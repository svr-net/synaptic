// Carried verbatim from svr-net/compute@c8b8594141f292c3a638e6735c0ff319d36ea76f (include/compute/parallel.hpp).
// Refresh with tools/sync_compute.sh; do not edit by hand.
/******************************************************************************************
    compute/parallel.hpp -- work over the CPU's cores, with no lock and no blocking wait

    `lanes(n, f)` runs f(0) .. f(n - 1) over the cores; `Once` runs a thing once, whichever thread
    asks first, the rest spinning until it is done; `Lazy` is a map filled from several threads at
    once, each entry made outside any critical section and published by compare-exchange. Each piece of work writes only its own
    output, so any order gives the same result -- as a GPU kernel's invocations do.

      * claiming   a lane takes the next index with one atomic fetch-and-add: no lock, no queue
      * failing    the first exception is kept by compare-exchange on a flag; the rest are
                   dropped, and the caller raises it once every lane is done
      * finishing  the caller works too, then spins on an atomic count of the lanes still
                   running (the CPU's pause, then a yield): no mutex, no condition variable,
                   no join. The lanes are detached: each touches nothing shared after it
                   counts itself out, and nothing is parked between calls, so an idle process
                   holds no thread and burns no cycle.

    One core in WASM, and when n or the machine has one.
******************************************************************************************/
#pragma once

#include <algorithm>
#include <atomic>
#include <cstddef>
#include <exception>
#include <functional>
#include <thread>
#include <utility>
#include <vector>

namespace compute {

/// One turn of a spin wait: the CPU told it is spinning (no syscall, no lock).
inline void relax() {
#if defined(__x86_64__) || defined(__i386__)
    __builtin_ia32_pause();
#elif defined(__aarch64__) || defined(__arm__)
    asm volatile("yield");
#endif
}

/// Spins until `done()` holds: a short run of pauses, then yields so a lane still working gets the core.
template <class Done>
void spin(Done&& done) {
    for (unsigned k = 0; !done(); ++k) {
        if (k < 64) relax();
        else std::this_thread::yield();
    }
}

/// Once, whichever thread comes first: the others spin until it is done (a compare-exchange claims it).
class Once {
public:
    template <class F>
    void operator()(F&& f) {
        int none = 0;
        if (state_.compare_exchange_strong(none, 1, std::memory_order_acq_rel)) {
            try {
                f();
            } catch (...) {
                state_.store(0, std::memory_order_release);   // not done: the next to ask tries again
                throw;
            }
            state_.store(2, std::memory_order_release);
            return;
        }
        spin([&] {
            int s = state_.load(std::memory_order_acquire);
            if (s == 0) (*this)(f);   // the first failed: this one tries
            return s != 1;
        });
    }

private:
    std::atomic<int> state_{0};   // 0 not yet, 1 running, 2 done
};

/// A map filled lazily from several threads at once, with no lock: an absent key's value is made by the thread that
/// asks (outside any critical section), then published by compare-exchange on its bucket's head; a thread that loses
/// the race to the same key drops its own and takes the one published. An entry never moves or changes once
/// published, so a reference to it stays good while the map holds it. Nothing is removed while threads use it:
/// `clear` and `erase_if` are for one thread alone, between the parallel parts.
template <class K, class V, class Hash = std::hash<K>>
class Lazy {
public:
    explicit Lazy(std::size_t buckets = 4096) : buckets_(buckets) {}
    ~Lazy() { clear(); }
    Lazy(const Lazy&) = delete;
    Lazy& operator=(const Lazy&) = delete;

    /// The entry of `k`, or nullptr.
    V* find(const K& k) const { return scan(bucket(k).load(std::memory_order_acquire), nullptr, k); }
    /// The entry of `k`, made by `make()` when absent.
    template <class Make>
    V& get(const K& k, Make&& make) {
        std::atomic<Node*>& b = bucket(k);
        Node* seen = b.load(std::memory_order_acquire);
        if (V* v = scan(seen, nullptr, k)) return *v;
        Node* n = new Node{k, make(), nullptr};
        for (Node* stop = nullptr;;) {
            if (V* v = scan(seen, stop, k)) {   // published meanwhile
                delete n;
                return *v;
            }
            n->next = seen;
            if (b.compare_exchange_weak(seen, n, std::memory_order_acq_rel, std::memory_order_acquire)) return n->value;
            stop = n->next;   // only what was published since needs scanning again
        }
    }
    bool contains(const K& k) const { return find(k) != nullptr; }

    /// One thread alone: every entry, or those `drop` names, removed.
    void clear() {
        erase_if([](const K&, const V&) { return true; });
    }
    template <class Drop>
    void erase_if(Drop&& drop) {
        for (auto& b : buckets_) {
            Node *keep = nullptr, *n = b.load(std::memory_order_relaxed);
            while (n) {
                Node* next = n->next;
                if (drop(n->key, n->value)) delete n;
                else n->next = keep, keep = n;
                n = next;
            }
            b.store(keep, std::memory_order_relaxed);
        }
    }

private:
    struct Node {
        K key;
        V value;
        Node* next;
    };
    std::atomic<Node*>& bucket(const K& k) const { return buckets_[Hash{}(k) % buckets_.size()]; }
    static V* scan(Node* n, const Node* stop, const K& k) {
        for (; n != stop; n = n->next)
            if (n->key == k) return &n->value;
        return nullptr;
    }
    mutable std::vector<std::atomic<Node*>> buckets_;
};

/// f(0) .. f(n - 1) over the CPU's cores (at most `most` of them, when given), each its own: work that writes only its
/// own slot gives the same result in any order. An exception in one is raised once all are done.
template <class F>
void lanes(std::size_t n, F&& f, std::size_t most = 0) {
#if defined(__EMSCRIPTEN__)
    (void)most;
    for (std::size_t i = 0; i < n; ++i) f(i);
#else
    std::size_t cores = std::max<std::size_t>(1, std::min<std::size_t>(n, std::thread::hardware_concurrency()));
    if (most) cores = std::min(cores, most);
    if (cores == 1) {
        for (std::size_t i = 0; i < n; ++i) f(i);
        return;
    }
    struct Shared {
        std::atomic<std::size_t> next{0};
        std::atomic<std::size_t> running{0};
        std::atomic<bool> failed{false};
        std::exception_ptr error;   // written once, by the lane that won `failed`
    } s;
    auto run = [&s, &f, n] {
        for (std::size_t i; (i = s.next.fetch_add(1, std::memory_order_relaxed)) < n;) {
            try {
                f(i);
            } catch (...) {
                bool first = false;
                if (s.failed.compare_exchange_strong(first, true, std::memory_order_acq_rel)) s.error = std::current_exception();
            }
        }
    };
    s.running.store(cores - 1, std::memory_order_relaxed);
    for (std::size_t c = 1; c < cores; ++c) {
        try {
            std::thread([&s, &run] {
                run();
                s.running.fetch_sub(1, std::memory_order_release);   // the last touch of anything shared
            }).detach();
        } catch (...) {   // no thread to be had: the caller takes its share
            s.running.fetch_sub(1, std::memory_order_relaxed);
        }
    }
    run();
    spin([&] { return s.running.load(std::memory_order_acquire) == 0; });
    if (s.failed.load(std::memory_order_relaxed)) std::rethrow_exception(s.error);
#endif
}

}  // namespace compute
