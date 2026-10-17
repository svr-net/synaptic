// The WGSL compiler of the software GPU: WGSL source -> bytecode (bytecode.hpp).
//
// Supported: the part of WGSL the library's kernels use.
//   module    struct, const, var<private | workgroup | uniform | storage, read | read_write>,
//             fn (one @compute entry point with @workgroup_size and @builtin parameters);
//             // and /* */ comments; @group(0) @binding(n)
//   types     bool, i32, u32, f32, vec2/3/4<T>, array<T, N>, array<T> (buffers), structs,
//             atomic<u32 | i32> (workgroup memory)
//   statements let, var, const, assignment (=, op=), ++/--, _ =, if / else, for, while, loop,
//             break, continue, return, function calls, blocks
//   expressions literals (abstract, u, i, f), the operators of WGSL with C precedence, swizzles,
//             indexing, constructors and conversions, user functions, and the builtins
//             select min max clamp abs sqrt inverseSqrt exp exp2 log log2 pow sin cos tan asin
//             acos atan atan2 sinh cosh tanh floor ceil round trunc fract sign dot length
//             distance normalize mix step workgroupBarrier storageBarrier atomicLoad
//             atomicStore atomicAdd atomicSub
// Every function is inlined into the entry point (WGSL has no recursion), so the program has
// one flat memory: [private globals | function slots | constants]. Constant expressions are
// folded with the machine's own scalar semantics.
#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdlib>
#include <deque>
#include <functional>
#include <map>
#include <memory>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

#include "bytecode.hpp"
#include "syn/gpu/software_gpu.hpp"

namespace syn::gpu {

namespace {

using namespace bc;

[[noreturn]] void error(int line, const std::string& msg) {
  throw std::runtime_error("WGSL:" + std::to_string(line) + ": " + msg);
}

// ================================================================ lexer

enum class Tk { Ident, Int, Float, Punct, End };

struct Token {
  Tk k = Tk::End;
  std::string s;
  std::uint64_t iv = 0;
  double fv = 0;
  char suffix = 0;
  int line = 0;
};

std::vector<Token> lex(const std::string& src) {
  static const char* puncts[] = {"<<=", ">>=", "->", "&&", "||", "==", "!=", "<=", ">=", "<<", ">>", "+=", "-=", "*=", "/=", "%=",
                                 "&=",  "|=",  "^=", "++", "--", "{",  "}",  "(",  ")",  "[",  "]",  "<",  ">",  ",",  ";",  ":",
                                 ".",   "=",   "+",  "-",  "*",  "/",  "%",  "&",  "|",  "^",  "~",  "!",  "@"};
  std::vector<Token> out;
  std::size_t i = 0;
  int line = 1;
  const std::size_t n = src.size();
  while (i < n) {
    const char c = src[i];
    if (c == '\n') {
      ++line;
      ++i;
      continue;
    }
    if (std::isspace(static_cast<unsigned char>(c))) {
      ++i;
      continue;
    }
    if (c == '/' && i + 1 < n && src[i + 1] == '/') {
      while (i < n && src[i] != '\n') ++i;
      continue;
    }
    if (c == '/' && i + 1 < n && src[i + 1] == '*') {
      int depth = 1;
      i += 2;
      while (i < n && depth > 0) {
        if (src[i] == '\n') ++line;
        if (src[i] == '/' && i + 1 < n && src[i + 1] == '*') {
          ++depth;
          i += 2;
        } else if (src[i] == '*' && i + 1 < n && src[i + 1] == '/') {
          --depth;
          i += 2;
        } else {
          ++i;
        }
      }
      continue;
    }
    Token t;
    t.line = line;
    if (std::isalpha(static_cast<unsigned char>(c)) || c == '_') {
      std::size_t j = i;
      while (j < n && (std::isalnum(static_cast<unsigned char>(src[j])) || src[j] == '_')) ++j;
      t.k = Tk::Ident;
      t.s = src.substr(i, j - i);
      i = j;
      out.push_back(t);
      continue;
    }
    if (std::isdigit(static_cast<unsigned char>(c)) || (c == '.' && i + 1 < n && std::isdigit(static_cast<unsigned char>(src[i + 1])))) {
      std::size_t j = i;
      if (c == '0' && j + 1 < n && (src[j + 1] == 'x' || src[j + 1] == 'X')) {
        j += 2;
        while (j < n && std::isxdigit(static_cast<unsigned char>(src[j]))) ++j;
        t.k = Tk::Int;
        t.iv = std::strtoull(src.substr(i + 2, j - i - 2).c_str(), nullptr, 16);
      } else {
        bool isFloat = false;
        while (j < n && std::isdigit(static_cast<unsigned char>(src[j]))) ++j;
        if (j < n && src[j] == '.') {
          isFloat = true;
          ++j;
          while (j < n && std::isdigit(static_cast<unsigned char>(src[j]))) ++j;
        }
        if (j < n && (src[j] == 'e' || src[j] == 'E')) {
          isFloat = true;
          ++j;
          if (j < n && (src[j] == '+' || src[j] == '-')) ++j;
          while (j < n && std::isdigit(static_cast<unsigned char>(src[j]))) ++j;
        }
        const std::string text = src.substr(i, j - i);
        if (j < n && (src[j] == 'f' || src[j] == 'h')) isFloat = true;
        if (isFloat) {
          t.k = Tk::Float;
          t.fv = std::strtod(text.c_str(), nullptr);
        } else {
          t.k = Tk::Int;
          t.iv = std::strtoull(text.c_str(), nullptr, 10);
        }
      }
      if (j < n && (src[j] == 'u' || src[j] == 'i' || src[j] == 'f' || src[j] == 'h')) t.suffix = src[j++];
      if (t.suffix == 'h') error(line, "f16 is not supported");
      t.s = src.substr(i, j - i);
      i = j;
      out.push_back(t);
      continue;
    }
    bool matched = false;
    for (const char* p : puncts) {
      const std::size_t len = std::char_traits<char>::length(p);
      if (src.compare(i, len, p) == 0) {
        t.k = Tk::Punct;
        t.s = p;
        i += len;
        out.push_back(t);
        matched = true;
        break;
      }
    }
    if (!matched) error(line, std::string("unexpected character '") + c + "'");
  }
  Token end;
  end.k = Tk::End;
  end.line = line;
  out.push_back(end);
  return out;
}

// ================================================================ syntax tree

struct Expr;
struct Stmt;
using ExprP = std::shared_ptr<Expr>;
using StmtP = std::shared_ptr<Stmt>;

struct TypeAst {
  std::string name;
  std::vector<TypeAst> args;
  ExprP count;  // array<T, count>
  int line = 0;
};

enum class Ek { Int, Float, Bool, Ident, Call, Member, Index, Unary, Binary, AddrOf };

struct Expr {
  Ek k = Ek::Int;
  int line = 0;
  std::string s;  // identifier, member, operator
  std::uint64_t iv = 0;
  double fv = 0;
  char suffix = 0;
  bool bv = false;
  TypeAst callee;  // Call: name and template arguments
  bool templated = false;
  std::vector<ExprP> args;
  ExprP a, b;
};

enum class Sk { Block, Let, Var, Const, Assign, IncDec, If, For, While, Loop, Break, Continue, Return, Call, Phony, Empty };

struct Stmt {
  Sk k = Sk::Empty;
  int line = 0;
  std::string name, op;
  TypeAst type;
  bool hasType = false;
  ExprP init, lhs, cond;
  std::vector<StmtP> body;
  StmtP then, els, first, update;
};

struct Attr {
  std::string name;
  std::vector<ExprP> args;
};

struct StructDecl {
  std::string name;
  std::vector<std::pair<std::string, TypeAst>> members;
  int line = 0;
};

struct ConstDecl {
  std::string name;
  TypeAst type;
  bool hasType = false;
  ExprP init;
  int line = 0;
};

struct GlobalVar {
  std::string name, space, access;
  TypeAst type;
  ExprP init;
  int group = -1, binding = -1;
  int line = 0;
};

struct Param {
  std::string name, builtin;
  TypeAst type;
};

struct FnDecl {
  std::string name;
  std::vector<Param> params;
  TypeAst ret;
  bool hasRet = false;
  StmtP body;
  bool compute = false;
  std::vector<ExprP> workgroupSize;
  int line = 0;
};

struct ModuleAst {
  std::vector<StructDecl> structs;
  std::vector<ConstDecl> consts;
  std::vector<GlobalVar> globals;
  std::vector<FnDecl> fns;
  std::vector<std::string> order;  // "s:", "c:", "g:", "f:" + name, in source order
};

// ================================================================ parser

class Parser {
 public:
  explicit Parser(std::vector<Token> toks) : t_(std::move(toks)) {}

  ModuleAst module() {
    ModuleAst m;
    while (peek().k != Tk::End) {
      std::vector<Attr> attrs = attributes();
      const Token& k = peek();
      if (isWord("struct")) {
        m.structs.push_back(structDecl());
        m.order.push_back("s:" + m.structs.back().name);
      } else if (isWord("const")) {
        m.consts.push_back(constDecl());
        m.order.push_back("c:" + m.consts.back().name);
      } else if (isWord("var")) {
        m.globals.push_back(globalVar(attrs));
        m.order.push_back("g:" + m.globals.back().name);
      } else if (isWord("fn")) {
        m.fns.push_back(fnDecl(attrs));
        m.order.push_back("f:" + m.fns.back().name);
      } else if (isWord("enable") || isWord("requires") || isWord("diagnostic")) {
        while (!isPunct(";")) next();
        next();
      } else if (isPunct(";")) {
        next();
      } else {
        error(k.line, "unexpected '" + k.s + "' at module scope");
      }
    }
    return m;
  }

 private:
  std::vector<Token> t_;
  std::size_t p_ = 0;

  const Token& peek(std::size_t ahead = 0) const { return t_[std::min(p_ + ahead, t_.size() - 1)]; }
  const Token& next() { return t_[p_ < t_.size() - 1 ? p_++ : p_]; }
  bool isPunct(const char* s, std::size_t ahead = 0) const { return peek(ahead).k == Tk::Punct && peek(ahead).s == s; }
  bool isWord(const char* s) const { return peek().k == Tk::Ident && peek().s == s; }
  bool accept(const char* s) {
    if (isPunct(s)) {
      next();
      return true;
    }
    return false;
  }
  void expect(const char* s) {
    if (!accept(s)) error(peek().line, std::string("expected '") + s + "', found '" + peek().s + "'");
  }
  // Closes a template list; splits '>>' and '>=' / '>>=' tokens.
  void expectGt() {
    Token& t = t_[p_];
    if (t.k == Tk::Punct && t.s == ">") {
      next();
      return;
    }
    if (t.k == Tk::Punct && (t.s == ">>" || t.s == ">=" || t.s == ">>=")) {
      t.s = t.s.substr(1);
      return;
    }
    error(t.line, "expected '>'");
  }
  std::string ident() {
    if (peek().k != Tk::Ident) error(peek().line, "expected an identifier, found '" + peek().s + "'");
    return next().s;
  }

  std::vector<Attr> attributes() {
    std::vector<Attr> out;
    while (accept("@")) {
      Attr a;
      a.name = ident();
      if (accept("(")) {
        if (!isPunct(")")) {
          a.args.push_back(expr());
          while (accept(",")) {
            if (isPunct(")")) break;
            a.args.push_back(expr());
          }
        }
        expect(")");
      }
      out.push_back(std::move(a));
    }
    return out;
  }

  static bool isTypeGenerator(const std::string& s) {
    return s == "vec2" || s == "vec3" || s == "vec4" || s == "array" || s == "atomic" || s == "ptr";
  }

  TypeAst type() {
    TypeAst t;
    t.line = peek().line;
    t.name = ident();
    if (isTypeGenerator(t.name) && accept("<")) {
      if (t.name == "ptr") error(t.line, "pointers are not supported");
      t.args.push_back(type());
      if (t.name == "array" && accept(",")) t.count = shift();
      expectGt();
    }
    return t;
  }

  StructDecl structDecl() {
    StructDecl s;
    s.line = peek().line;
    next();
    s.name = ident();
    expect("{");
    while (!accept("}")) {
      attributes();
      const std::string name = ident();
      expect(":");
      s.members.emplace_back(name, type());
      if (!accept(",")) accept(";");
    }
    accept(";");
    return s;
  }

  ConstDecl constDecl() {
    ConstDecl c;
    c.line = peek().line;
    next();
    c.name = ident();
    if (accept(":")) {
      c.type = type();
      c.hasType = true;
    }
    expect("=");
    c.init = expr();
    expect(";");
    return c;
  }

  GlobalVar globalVar(const std::vector<Attr>& attrs) {
    GlobalVar g;
    g.line = peek().line;
    next();
    g.space = "private";
    if (accept("<")) {
      g.space = ident();
      if (accept(",")) g.access = ident();
      expectGt();
    }
    g.name = ident();
    expect(":");
    g.type = type();
    if (accept("=")) g.init = expr();
    expect(";");
    for (const auto& a : attrs) {
      if (a.name == "group" && !a.args.empty()) g.group = static_cast<int>(a.args[0]->iv);
      if (a.name == "binding" && !a.args.empty()) g.binding = static_cast<int>(a.args[0]->iv);
    }
    return g;
  }

  FnDecl fnDecl(const std::vector<Attr>& attrs) {
    FnDecl f;
    f.line = peek().line;
    next();
    f.name = ident();
    for (const auto& a : attrs) {
      if (a.name == "compute") f.compute = true;
      if (a.name == "workgroup_size") f.workgroupSize = a.args;
    }
    expect("(");
    while (!accept(")")) {
      std::vector<Attr> pa = attributes();
      Param p;
      p.name = ident();
      expect(":");
      p.type = type();
      for (const auto& a : pa)
        if (a.name == "builtin" && !a.args.empty()) p.builtin = a.args[0]->s;
      f.params.push_back(std::move(p));
      if (!accept(",")) {
        expect(")");
        break;
      }
    }
    if (accept("->")) {
      attributes();
      f.ret = type();
      f.hasRet = true;
    }
    f.body = block();
    return f;
  }

  // ---- statements

  StmtP block() {
    auto s = std::make_shared<Stmt>();
    s->k = Sk::Block;
    s->line = peek().line;
    expect("{");
    while (!accept("}")) {
      if (peek().k == Tk::End) error(s->line, "unterminated block");
      s->body.push_back(statement());
    }
    return s;
  }

  StmtP declaration(Sk kind) {
    auto s = std::make_shared<Stmt>();
    s->k = kind;
    s->line = peek().line;
    next();
    if (kind == Sk::Var && accept("<")) {
      ident();
      expectGt();
    }
    s->name = ident();
    if (accept(":")) {
      s->type = type();
      s->hasType = true;
    }
    if (accept("=")) s->init = expr();
    if (kind != Sk::Var && !s->init) error(s->line, "'" + s->name + "' needs an initializer");
    return s;
  }

  // A statement without its trailing ';' (for headers of for loops, too).
  StmtP simple() {
    if (isWord("let")) return declaration(Sk::Let);
    if (isWord("var")) return declaration(Sk::Var);
    if (isWord("const")) return declaration(Sk::Const);
    auto s = std::make_shared<Stmt>();
    s->line = peek().line;
    if (isWord("_") && isPunct("=", 1)) {
      next();
      next();
      s->k = Sk::Phony;
      s->init = expr();
      return s;
    }
    ExprP e = unary();
    static const char* assigns[] = {"=", "+=", "-=", "*=", "/=", "%=", "&=", "|=", "^=", "<<=", ">>="};
    for (const char* op : assigns)
      if (isPunct(op)) {
        next();
        s->k = Sk::Assign;
        s->op = op;
        s->lhs = e;
        s->init = expr();
        return s;
      }
    if (isPunct("++") || isPunct("--")) {
      s->k = Sk::IncDec;
      s->op = next().s;
      s->lhs = e;
      return s;
    }
    if (e->k != Ek::Call) error(s->line, "expected a statement");
    s->k = Sk::Call;
    s->init = e;
    return s;
  }

  StmtP body() { return isPunct("{") ? block() : statement(); }

  StmtP statement() {
    const int line = peek().line;
    if (isPunct("{")) return block();
    if (accept(";")) {
      auto s = std::make_shared<Stmt>();
      s->k = Sk::Empty;
      return s;
    }
    auto s = std::make_shared<Stmt>();
    s->line = line;
    if (isWord("if")) {
      next();
      s->k = Sk::If;
      s->cond = expr();
      s->then = body();
      if (isWord("else")) {
        next();
        s->els = isWord("if") ? statement() : body();
      }
      return s;
    }
    if (isWord("for")) {
      next();
      s->k = Sk::For;
      expect("(");
      if (!isPunct(";")) s->first = simple();
      expect(";");
      if (!isPunct(";")) s->cond = expr();
      expect(";");
      if (!isPunct(")")) s->update = simple();
      expect(")");
      s->then = body();
      return s;
    }
    if (isWord("while")) {
      next();
      s->k = Sk::While;
      s->cond = expr();
      s->then = body();
      return s;
    }
    if (isWord("loop")) {
      next();
      s->k = Sk::Loop;
      s->then = block();
      return s;
    }
    if (isWord("break")) {
      next();
      s->k = Sk::Break;
      expect(";");
      return s;
    }
    if (isWord("continue")) {
      next();
      s->k = Sk::Continue;
      expect(";");
      return s;
    }
    if (isWord("return")) {
      next();
      s->k = Sk::Return;
      if (!isPunct(";")) s->init = expr();
      expect(";");
      return s;
    }
    StmtP x = simple();
    expect(";");
    return x;
  }

  // ---- expressions (C precedence; valid WGSL never relies on the differences)

  ExprP binaryNode(const std::string& op, ExprP a, ExprP b, int line) {
    auto e = std::make_shared<Expr>();
    e->k = Ek::Binary;
    e->s = op;
    e->a = std::move(a);
    e->b = std::move(b);
    e->line = line;
    return e;
  }

  ExprP expr() { return logicalOr(); }
  ExprP logicalOr() {
    ExprP e = logicalAnd();
    while (isPunct("||")) {
      const int line = next().line;
      e = binaryNode("||", e, logicalAnd(), line);
    }
    return e;
  }
  ExprP logicalAnd() {
    ExprP e = bitOr();
    while (isPunct("&&")) {
      const int line = next().line;
      e = binaryNode("&&", e, bitOr(), line);
    }
    return e;
  }
  ExprP bitOr() {
    ExprP e = bitXor();
    while (isPunct("|")) {
      const int line = next().line;
      e = binaryNode("|", e, bitXor(), line);
    }
    return e;
  }
  ExprP bitXor() {
    ExprP e = bitAnd();
    while (isPunct("^")) {
      const int line = next().line;
      e = binaryNode("^", e, bitAnd(), line);
    }
    return e;
  }
  ExprP bitAnd() {
    ExprP e = equality();
    while (isPunct("&")) {
      const int line = next().line;
      e = binaryNode("&", e, equality(), line);
    }
    return e;
  }
  ExprP equality() {
    ExprP e = relational();
    while (isPunct("==") || isPunct("!=")) {
      const Token& t = next();
      e = binaryNode(t.s, e, relational(), t.line);
    }
    return e;
  }
  ExprP relational() {
    ExprP e = shift();
    while (isPunct("<") || isPunct("<=") || isPunct(">") || isPunct(">=")) {
      const Token& t = next();
      e = binaryNode(t.s, e, shift(), t.line);
    }
    return e;
  }
  ExprP shift() {
    ExprP e = additive();
    while (isPunct("<<") || isPunct(">>")) {
      const Token& t = next();
      e = binaryNode(t.s, e, additive(), t.line);
    }
    return e;
  }
  ExprP additive() {
    ExprP e = multiplicative();
    while (isPunct("+") || isPunct("-")) {
      const Token& t = next();
      e = binaryNode(t.s, e, multiplicative(), t.line);
    }
    return e;
  }
  ExprP multiplicative() {
    ExprP e = unary();
    while (isPunct("*") || isPunct("/") || isPunct("%")) {
      const Token& t = next();
      e = binaryNode(t.s, e, unary(), t.line);
    }
    return e;
  }
  ExprP unary() {
    if (isPunct("-") || isPunct("!") || isPunct("~") || isPunct("&")) {
      const Token& t = next();
      auto e = std::make_shared<Expr>();
      e->k = t.s == "&" ? Ek::AddrOf : Ek::Unary;
      e->s = t.s;
      e->line = t.line;
      e->a = unary();
      return e;
    }
    return postfix(primary());
  }
  ExprP postfix(ExprP e) {
    for (;;) {
      if (isPunct(".")) {
        const int line = next().line;
        auto m = std::make_shared<Expr>();
        m->k = Ek::Member;
        m->line = line;
        m->a = e;
        m->s = ident();
        e = m;
      } else if (isPunct("[")) {
        const int line = next().line;
        auto m = std::make_shared<Expr>();
        m->k = Ek::Index;
        m->line = line;
        m->a = e;
        m->b = expr();
        expect("]");
        e = m;
      } else {
        return e;
      }
    }
  }
  ExprP primary() {
    const Token& t = peek();
    auto e = std::make_shared<Expr>();
    e->line = t.line;
    if (t.k == Tk::Int || t.k == Tk::Float) {
      next();
      e->k = t.k == Tk::Int ? Ek::Int : Ek::Float;
      e->iv = t.iv;
      e->fv = t.k == Tk::Int ? static_cast<double>(t.iv) : t.fv;
      e->suffix = t.suffix;
      if (t.k == Tk::Int && t.suffix == 'f') {
        e->k = Ek::Float;
        e->fv = static_cast<double>(t.iv);
      }
      return e;
    }
    if (accept("(")) {
      ExprP inner = expr();
      expect(")");
      return inner;
    }
    if (t.k == Tk::Ident) {
      if (t.s == "true" || t.s == "false") {
        next();
        e->k = Ek::Bool;
        e->bv = t.s == "true";
        return e;
      }
      if (isTypeGenerator(t.s) && isPunct("<", 1)) {
        e->k = Ek::Call;
        e->callee = type();
        e->templated = true;
      } else if (isPunct("(", 1)) {
        e->k = Ek::Call;
        e->callee.name = next().s;
        e->callee.line = t.line;
      } else {
        e->k = Ek::Ident;
        e->s = next().s;
        return e;
      }
      expect("(");
      if (!isPunct(")")) {
        e->args.push_back(expr());
        while (accept(",")) {
          if (isPunct(")")) break;
          e->args.push_back(expr());
        }
      }
      expect(")");
      return e;
    }
    error(t.line, "unexpected '" + t.s + "'");
  }
};

// ================================================================ types

enum class K { Void, Bool, I32, U32, F32, AInt, AFloat, Vec, Arr, Struct, Atomic };

struct Ty {
  K k = K::Void;
  int n = 0;  // vector size, array count (-1: runtime-sized)
  const Ty* el = nullptr;
  int sid = -1;
  std::uint32_t words = 0;
};

struct Member {
  std::string name;
  const Ty* t;
  std::uint32_t offset;
};

struct StructInfo {
  std::string name;
  std::vector<Member> members;
};

bool isScalar(const Ty* t) { return t->k == K::Bool || t->k == K::I32 || t->k == K::U32 || t->k == K::F32 || t->k == K::AInt || t->k == K::AFloat; }
bool isAbstract(const Ty* t) { return t->k == K::AInt || t->k == K::AFloat; }

// ================================================================ code generation

constexpr std::uint32_t kConstTag = 0x80000000u;

struct Val {
  const Ty* t = nullptr;
  bool isConst = false;
  std::vector<std::uint32_t> w;  // concrete constant words
  double af = 0;                 // abstract float
  std::int64_t ai = 0;           // abstract int
  std::uint32_t slot = 0;
  bool temp = false;       // a fresh temporary
  bool immutable = false;  // never changes while visible (temporaries, lets, parameters)
  bool priv = false;       // aliases a private global
};

struct Ref {
  enum Space { Mem, Wg, Buf } sp = Mem;
  const Ty* t = nullptr;
  std::uint32_t base = 0;
  std::uint32_t binding = 0;
  std::int64_t dyn = -1;
  bool writable = true;
  bool priv = false;
};

struct Binding {
  enum Kind { Const, Let, Var, Global } kind = Const;
  Val v;
  Ref r;
};

class Compiler {
 public:
  Compiler(ModuleAst m, Shader& out) : m_(std::move(m)), out_(out) {}

  void compile(const std::string& entry) {
    for (const auto& item : m_.order) {
      const char kind = item[0];
      const std::string nm = item.substr(2);
      if (kind == 's') declareStruct(nm);
      if (kind == 'c') declareConst(nm);
      if (kind == 'g') declareGlobal(nm);
      if (kind == 'f')
        for (const auto& fn : m_.fns)
          if (fn.name == nm) fns_[nm] = &fn;
    }
    auto it = fns_.find(entry);
    if (it == fns_.end() || !it->second->compute) error(0, "no @compute entry point '" + entry + "'");
    const FnDecl& f = *it->second;
    for (int i = 0; i < 3; ++i) out_.workgroupSize[i] = 1;
    for (std::size_t i = 0; i < f.workgroupSize.size() && i < 3; ++i) {
      Val v = expr(*f.workgroupSize[i]);
      if (!v.isConst) error(f.line, "@workgroup_size must be constant");
      out_.workgroupSize[i] = static_cast<std::uint32_t>(isAbstract(v.t) ? v.ai : v.w[0]);
    }
    next_ = privateWords_;
    maxNext_ = next_;
    scopes_.assign(1, {});
    for (const auto& p : f.params) {
      const Ty* t = resolve(p.type);
      const std::uint32_t slot = alloc(t->words);
      Shader::Builtin b;
      if (p.builtin == "local_invocation_id") b = Shader::LocalId;
      else if (p.builtin == "workgroup_id") b = Shader::WorkgroupId;
      else if (p.builtin == "global_invocation_id") b = Shader::GlobalId;
      else if (p.builtin == "num_workgroups") b = Shader::NumWorkgroups;
      else if (p.builtin == "local_invocation_index") b = Shader::LocalIndex;
      else error(f.line, "unsupported entry parameter '" + p.name + "'");
      out_.builtins.push_back({b, slot});
      Binding bd;
      bd.kind = Binding::Let;
      bd.v.t = t;
      bd.v.slot = slot;
      bd.v.immutable = true;
      scopes_.back()[p.name] = bd;
    }
    for (const auto& [gname, init] : privateInits_) {
      Binding& g = globals_[gname];
      store(g.r, expr(*init), init->line);
    }
    rets_.clear();
    stmt(*f.body);
    emit(END);
    finish();
  }

 private:
  ModuleAst m_;
  Shader& out_;
  std::deque<Ty> types_;
  std::vector<StructInfo> structs_;
  std::unordered_map<std::string, int> structIds_;
  std::unordered_map<std::string, Binding> globals_;
  std::unordered_map<std::string, const FnDecl*> fns_;
  std::vector<std::pair<std::string, ExprP>> privateInits_;
  std::uint32_t privateWords_ = 0, workgroupWords_ = 0;
  std::vector<Ins>& code_ = out_.code;
  std::uint32_t next_ = 0, maxNext_ = 0;
  std::vector<std::uint32_t> pool_;
  std::map<std::vector<std::uint32_t>, std::uint32_t> poolIndex_;
  std::vector<std::unordered_map<std::string, Binding>> scopes_;
  struct Loop {
    std::vector<std::size_t> breaks, continues;
  };
  std::vector<Loop> loops_;
  struct Ret {
    const Ty* t;
    std::uint32_t slot;
    std::vector<std::size_t> jumps;
  };
  std::vector<Ret> rets_;
  int depth_ = 0;
  std::size_t lastLabel_ = static_cast<std::size_t>(-1);

  // ---------------------------------------------------------------- types

  const Ty* intern(const Ty& t) {
    for (const auto& x : types_)
      if (x.k == t.k && x.n == t.n && x.el == t.el && x.sid == t.sid) return &x;
    types_.push_back(t);
    return &types_.back();
  }
  const Ty* scalar(K k) {
    Ty t;
    t.k = k;
    t.words = k == K::Void ? 0 : 1;
    return intern(t);
  }
  const Ty* vec(int n, const Ty* el) {
    Ty t;
    t.k = K::Vec;
    t.n = n;
    t.el = el;
    t.words = static_cast<std::uint32_t>(n);
    return intern(t);
  }
  const Ty* arr(const Ty* el, int n) {
    Ty t;
    t.k = K::Arr;
    t.n = n;
    t.el = el;
    t.words = n < 0 ? 0 : static_cast<std::uint32_t>(n) * el->words;
    return intern(t);
  }
  const Ty* tF32() { return scalar(K::F32); }
  const Ty* tU32() { return scalar(K::U32); }
  const Ty* tI32() { return scalar(K::I32); }
  const Ty* tBool() { return scalar(K::Bool); }

  std::string name(const Ty* t) const {
    switch (t->k) {
      case K::Void: return "void";
      case K::Bool: return "bool";
      case K::I32: return "i32";
      case K::U32: return "u32";
      case K::F32: return "f32";
      case K::AInt: return "abstract-int";
      case K::AFloat: return "abstract-float";
      case K::Vec: return "vec" + std::to_string(t->n) + "<" + name(t->el) + ">";
      case K::Arr: return "array<" + name(t->el) + (t->n >= 0 ? ", " + std::to_string(t->n) : "") + ">";
      case K::Struct: return structs_[static_cast<std::size_t>(t->sid)].name;
      case K::Atomic: return "atomic<" + name(t->el) + ">";
    }
    return "?";
  }

  const Ty* resolve(const TypeAst& a) {
    const std::string& n = a.name;
    if (n == "f32") return tF32();
    if (n == "u32") return tU32();
    if (n == "i32") return tI32();
    if (n == "bool") return tBool();
    if (n == "vec2" || n == "vec3" || n == "vec4") {
      if (a.args.empty()) error(a.line, n + " needs an element type");
      return vec(n[3] - '0', resolve(a.args[0]));
    }
    if (n == "vec2f" || n == "vec3f" || n == "vec4f") return vec(n[3] - '0', tF32());
    if (n == "vec2u" || n == "vec3u" || n == "vec4u") return vec(n[3] - '0', tU32());
    if (n == "vec2i" || n == "vec3i" || n == "vec4i") return vec(n[3] - '0', tI32());
    if (n == "array") {
      if (a.args.empty()) error(a.line, "array needs an element type");
      const Ty* el = resolve(a.args[0]);
      if (!a.count) return arr(el, -1);
      Val c = expr(*a.count);
      if (!c.isConst) error(a.line, "array size must be constant");
      const std::int64_t count = isAbstract(c.t) ? c.ai : static_cast<std::int64_t>(c.w[0]);
      if (count <= 0) error(a.line, "array size must be positive");
      return arr(el, static_cast<int>(count));
    }
    if (n == "atomic") {
      Ty t;
      t.k = K::Atomic;
      t.el = resolve(a.args.at(0));
      t.words = 1;
      return intern(t);
    }
    auto it = structIds_.find(n);
    if (it != structIds_.end()) {
      Ty t;
      t.k = K::Struct;
      t.sid = it->second;
      for (const auto& mm : structs_[static_cast<std::size_t>(it->second)].members) t.words += mm.t->words;
      return intern(t);
    }
    error(a.line, "unknown type '" + n + "'");
  }

  bool isTypeName(const std::string& n) const {
    return n == "f32" || n == "u32" || n == "i32" || n == "bool" || n == "vec2" || n == "vec3" || n == "vec4" || n == "array" ||
           n == "vec2f" || n == "vec3f" || n == "vec4f" || n == "vec2u" || n == "vec3u" || n == "vec4u" || n == "vec2i" ||
           n == "vec3i" || n == "vec4i" || structIds_.count(n) > 0;
  }

  // ---------------------------------------------------------------- module declarations

  void declareStruct(const std::string& nm) {
    for (const auto& s : m_.structs)
      if (s.name == nm) {
        StructInfo info;
        info.name = nm;
        std::uint32_t off = 0;
        for (const auto& [mn, ta] : s.members) {
          const Ty* t = resolve(ta);
          info.members.push_back({mn, t, off});
          off += t->words;
        }
        structIds_[nm] = static_cast<int>(structs_.size());
        structs_.push_back(std::move(info));
        return;
      }
  }

  void declareConst(const std::string& nm) {
    for (const auto& c : m_.consts)
      if (c.name == nm) {
        Val v = expr(*c.init);
        if (!v.isConst) error(c.line, "const '" + nm + "' is not a constant expression");
        if (c.hasType) v = convert(v, resolve(c.type), c.line);
        Binding b;
        b.kind = Binding::Const;
        b.v = v;
        globals_[nm] = b;
        return;
      }
  }

  void declareGlobal(const std::string& nm) {
    for (const auto& g : m_.globals)
      if (g.name == nm) {
        const Ty* t = resolve(g.type);
        Binding b;
        b.kind = Binding::Global;
        b.r.t = t;
        if (g.space == "private") {
          b.r.sp = Ref::Mem;
          b.r.base = privateWords_;
          b.r.priv = true;
          privateWords_ += t->words;
          if (g.init) privateInits_.emplace_back(nm, g.init);
        } else if (g.space == "workgroup") {
          b.r.sp = Ref::Wg;
          b.r.base = workgroupWords_;
          workgroupWords_ += t->words;
        } else if (g.space == "uniform" || g.space == "storage") {
          if (g.binding < 0) error(g.line, "'" + nm + "' needs @binding");
          if (g.group > 0) error(g.line, "only @group(0) is supported");
          b.r.sp = Ref::Buf;
          b.r.binding = static_cast<std::uint32_t>(g.binding);
          b.r.writable = g.space == "storage" && g.access == "read_write";
          BindingInfo info;
          info.binding = b.r.binding;
          info.uniform = g.space == "uniform";
          info.writable = b.r.writable;
          info.name = nm;
          out_.bindings.push_back(info);
        } else {
          error(g.line, "unsupported address space '" + g.space + "'");
        }
        globals_[nm] = b;
        return;
      }
  }

  // ---------------------------------------------------------------- emission

  std::size_t emit(std::uint32_t op, std::uint32_t a = 0, std::uint32_t b = 0, std::uint32_t c = 0, std::uint32_t d = 0, std::uint32_t e = 0) {
    Ins i;
    i.op = op;
    i.a = a;
    i.b = b;
    i.c = c;
    i.d = d;
    i.e = e;
    code_.push_back(i);
    return code_.size() - 1;
  }
  std::uint32_t here() {
    lastLabel_ = code_.size();
    return static_cast<std::uint32_t>(code_.size());
  }
  void patch(std::size_t at, std::uint32_t target) {
    Ins& i = code_[at];
    if (i.op == JMP) i.a = target;
    else i.b = target;
  }

  std::uint32_t alloc(std::uint32_t words) {
    const std::uint32_t s = next_;
    next_ += words;
    if (next_ > maxNext_) maxNext_ = next_;
    if (next_ >= kConstTag) error(0, "out of memory");
    return s;
  }

  std::uint32_t constAddr(const std::vector<std::uint32_t>& words) {
    auto it = poolIndex_.find(words);
    if (it != poolIndex_.end()) return kConstTag | it->second;
    const auto idx = static_cast<std::uint32_t>(pool_.size());
    pool_.insert(pool_.end(), words.begin(), words.end());
    poolIndex_[words] = idx;
    return kConstTag | idx;
  }
  std::uint32_t constU32(std::uint32_t v) { return constAddr({v}); }
  std::uint32_t zeroWords(std::uint32_t n) { return constAddr(std::vector<std::uint32_t>(n, 0u)); }

  // Removes jumps to the next instruction and threads jumps to jumps.
  void tidyJumps() {
    auto target = [&](const Ins& i) -> std::uint32_t { return i.op == JMP ? i.a : i.b; };
    auto setTarget = [&](Ins& i, std::uint32_t t) {
      if (i.op == JMP) i.a = t;
      else i.b = t;
    };
    const auto n = static_cast<std::uint32_t>(code_.size());
    auto final = [&](std::uint32_t t) {
      for (int hops = 0; hops < 64 && t < n && code_[t].op == JMP; ++hops) t = code_[t].a;
      return t;
    };
    for (auto& i : code_)
      if (i.op == JMP || i.op == JZ || i.op == JNZ) setTarget(i, final(target(i)));
    std::vector<std::uint32_t> map(n + 1);
    std::vector<bool> keep(n, true);
    for (std::uint32_t k = 0; k < n; ++k)
      if (code_[k].op == JMP && code_[k].a == k + 1) keep[k] = false;
    std::uint32_t out = 0;
    for (std::uint32_t k = 0; k < n; ++k) {
      map[k] = out;
      if (keep[k]) ++out;
    }
    map[n] = out;
    std::vector<Ins> code;
    code.reserve(out);
    for (std::uint32_t k = 0; k < n; ++k)
      if (keep[k]) {
        Ins i = code_[k];
        if (i.op == JMP || i.op == JZ || i.op == JNZ) setTarget(i, map[target(i)]);
        code.push_back(i);
      }
    code_.swap(code);
  }

  void finish() {
    tidyJumps();
    const std::uint32_t base = maxNext_;
    for (auto& i : code_) {
      const char* r = roles(i.op);
      std::uint32_t* f[5] = {&i.a, &i.b, &i.c, &i.d, &i.e};
      for (int k = 0; k < 5; ++k)
        if (r[k] == 'r' && (*f[k] & kConstTag)) *f[k] = base + (*f[k] & ~kConstTag);
    }
    // The zero operand of offset-free memory instructions may also be a constant: relocated above.
    out_.privateWords = privateWords_;
    out_.constantBase = base;
    out_.constants = pool_;
    out_.memoryWords = base + static_cast<std::uint32_t>(pool_.size()) + 1;
    out_.workgroupWords = workgroupWords_;
  }

  // ---------------------------------------------------------------- values

  Val constant(const Ty* t, std::vector<std::uint32_t> w) {
    Val v;
    v.t = t;
    v.isConst = true;
    v.w = std::move(w);
    v.immutable = true;
    return v;
  }
  Val abstractInt(std::int64_t x) {
    Val v;
    v.t = scalar(K::AInt);
    v.isConst = true;
    v.ai = x;
    v.af = static_cast<double>(x);
    v.immutable = true;
    return v;
  }
  Val abstractFloat(double x) {
    Val v;
    v.t = scalar(K::AFloat);
    v.isConst = true;
    v.af = x;
    v.immutable = true;
    return v;
  }
  Val voidVal() {
    Val v;
    v.t = scalar(K::Void);
    return v;
  }
  Val tempOf(const Ty* t) {
    Val v;
    v.t = t;
    v.slot = alloc(t->words);
    v.temp = true;
    v.immutable = true;
    return v;
  }

  // The concrete type an abstract value takes by default.
  Val concretize(const Val& v) {
    if (v.t->k == K::AInt) return constant(tI32(), {static_cast<std::uint32_t>(static_cast<std::int32_t>(v.ai))});
    if (v.t->k == K::AFloat) return constant(tF32(), {W(static_cast<float>(v.af))});
    return v;
  }

  // Implicit conversion (abstract to concrete); anything else must already match.
  Val convert(const Val& v, const Ty* to, int line) {
    if (v.t == to) return v;
    if (v.t->k == K::AInt) {
      if (to->k == K::I32) return constant(to, {static_cast<std::uint32_t>(static_cast<std::int32_t>(v.ai))});
      if (to->k == K::U32) {
        if (v.ai < 0) error(line, "negative value for u32");
        return constant(to, {static_cast<std::uint32_t>(v.ai)});
      }
      if (to->k == K::F32) return constant(to, {W(static_cast<float>(v.ai))});
      if (to->k == K::AFloat) return abstractFloat(static_cast<double>(v.ai));
    }
    if (v.t->k == K::AFloat && to->k == K::F32) return constant(to, {W(static_cast<float>(v.af))});
    error(line, "cannot convert " + name(v.t) + " to " + name(to));
  }

  std::uint32_t slotOf(const Val& v) {
    if (!v.isConst) return v.slot;
    return constAddr(concretize(v).w);
  }

  // The scalar type of a value's components.
  static const Ty* elemOf(const Ty* t) { return t->k == K::Vec ? t->el : t; }

  // Unifies the abstract operands of an operation with the concrete one.
  void unify(Val& a, Val& b, int line) {
    const Ty* ea = elemOf(a.t);
    const Ty* eb = elemOf(b.t);
    if (isAbstract(ea) && !isAbstract(eb)) a = convert(a, a.t->k == K::Vec ? vec(a.t->n, eb) : eb, line);
    else if (isAbstract(eb) && !isAbstract(ea)) b = convert(b, b.t->k == K::Vec ? vec(b.t->n, ea) : ea, line);
    else if (ea->k == K::AInt && eb->k == K::AFloat) a = abstractFloat(static_cast<double>(a.ai));
    else if (eb->k == K::AInt && ea->k == K::AFloat) b = abstractFloat(static_cast<double>(b.ai));
  }

  // Applies a scalar operation componentwise (broadcasting scalars), folding constants.
  Val componentwise(std::uint32_t op, const Ty* result, const std::vector<Val>& args) {
    const int n = result->k == K::Vec ? result->n : 1;
    bool allConst = true;
    for (const auto& a : args) allConst = allConst && a.isConst;
    auto comp = [&](const Val& a, int i) -> std::uint32_t {
      const Val c = concretize(a);
      return a.t->k == K::Vec ? c.w[static_cast<std::size_t>(i)] : c.w[0];
    };
    if (allConst) {
      std::vector<std::uint32_t> w;
      for (int i = 0; i < n; ++i) {
        if (op == SEL) w.push_back(comp(args[2], i) != 0u ? comp(args[1], i) : comp(args[0], i));
        else if (args.size() == 1) w.push_back(unary(op, comp(args[0], i)));
        else w.push_back(binary(op, comp(args[0], i), comp(args[1], i)));
      }
      return constant(result, w);
    }
    Val r = tempOf(result);
    std::vector<std::uint32_t> base;
    for (const auto& a : args) base.push_back(slotOf(a));
    for (int i = 0; i < n; ++i) {
      auto at = [&](std::size_t k) { return args[k].t->k == K::Vec ? base[k] + static_cast<std::uint32_t>(i) : base[k]; };
      const std::uint32_t d = r.slot + static_cast<std::uint32_t>(i);
      if (op == SEL) emit(SEL, d, at(0), at(1), at(2));
      else if (args.size() == 1) emit(op, d, at(0));
      else emit(op, d, at(0), at(1));
    }
    return r;
  }

  // ---------------------------------------------------------------- scopes

  const Binding* lookup(const std::string& n) const {
    for (auto it = scopes_.rbegin(); it != scopes_.rend(); ++it) {
      auto f = it->find(n);
      if (f != it->end()) return &f->second;
    }
    auto g = globals_.find(n);
    return g == globals_.end() ? nullptr : &g->second;
  }

  // ---------------------------------------------------------------- references

  struct Place {
    bool isRef = false;
    Ref r;
    Val v;
  };

  Val load(const Ref& r) {
    const std::uint32_t n = r.t->words;
    if (r.sp == Ref::Mem && r.dyn < 0) {
      Val v;
      v.t = r.t;
      v.slot = r.base;
      v.priv = r.priv;
      v.immutable = !r.writable;
      return v;
    }
    Val t = tempOf(r.t);
    const std::uint32_t off = r.dyn >= 0 ? static_cast<std::uint32_t>(r.dyn) : constU32(0);
    if (r.sp == Ref::Mem) emit(LDM, t.slot, r.base, off, n);
    else if (r.sp == Ref::Wg) emit(LDW, t.slot, r.base, off, n);
    else emit(LDB, t.slot, r.binding, off, n, r.base);
    return t;
  }

  bool lastWrites(std::uint32_t slot) const {
    if (code_.empty() || lastLabel_ == code_.size()) return false;
    const Ins& i = code_.back();
    if (i.a != slot) return false;
    if (isUnary(i.op) || isBinary(i.op) || i.op == SEL || i.op == MOV) return true;
    if ((i.op == LDM || i.op == LDW || i.op == LDB) && i.d == 1) return true;
    return i.op == ATLD;
  }

  void store(const Ref& r, Val v, int line) {
    if (!r.writable) error(line, "cannot assign to a read-only value");
    if (r.t->k == K::Atomic) error(line, "use atomicStore for atomics");
    v = convert(v, r.t, line);
    const std::uint32_t n = r.t->words;
    if (r.sp == Ref::Mem && r.dyn < 0) {
      if (!v.isConst && v.slot == r.base) return;
      if (n == 1 && !v.isConst && v.temp && lastWrites(v.slot)) {
        code_.back().a = r.base;
        return;
      }
      const std::uint32_t src = slotOf(v);
      if (n == 1) emit(MOV, r.base, src);
      else emit(MOVN, r.base, src, n);
      return;
    }
    const std::uint32_t src = slotOf(v);
    const std::uint32_t off = r.dyn >= 0 ? static_cast<std::uint32_t>(r.dyn) : constU32(0);
    if (r.sp == Ref::Mem) emit(STM, r.base, off, src, n);
    else if (r.sp == Ref::Wg) emit(STW, r.base, off, src, n);
    else emit(STB, r.binding, off, src, n, r.base);
  }

  static int swizzleIndex(char c) {
    switch (c) {
      case 'x': case 'r': return 0;
      case 'y': case 'g': return 1;
      case 'z': case 'b': return 2;
      case 'w': case 'a': return 3;
      default: return -1;
    }
  }

  Place place(const Expr& e) {
    Place p;
    if (e.k == Ek::Ident) {
      const Binding* b = lookup(e.s);
      if (!b) error(e.line, "unknown identifier '" + e.s + "'");
      if (b->kind == Binding::Var || b->kind == Binding::Global) {
        p.isRef = true;
        p.r = b->r;
      } else {
        p.v = b->v;
      }
      return p;
    }
    if (e.k == Ek::Member || e.k == Ek::Index) {
      Place base = place(*e.a);
      const Ty* t = base.isRef ? base.r.t : base.v.t;
      if (e.k == Ek::Member) {
        std::uint32_t offset = 0;
        const Ty* mt = nullptr;
        if (t->k == K::Struct) {
          for (const auto& mm : structs_[static_cast<std::size_t>(t->sid)].members)
            if (mm.name == e.s) {
              offset = mm.offset;
              mt = mm.t;
            }
          if (!mt) error(e.line, name(t) + " has no member '" + e.s + "'");
        } else if (t->k == K::Vec) {
          if (e.s.size() > 1) {  // a multi-component swizzle is a new value
            Val src = base.isRef ? load(base.r) : base.v;
            std::vector<int> idx;
            for (char c : e.s) {
              const int k = swizzleIndex(c);
              if (k < 0 || k >= t->n) error(e.line, "bad swizzle ." + e.s);
              idx.push_back(k);
            }
            const Ty* rt = vec(static_cast<int>(idx.size()), t->el);
            if (src.isConst) {
              Val c = concretize(src);
              std::vector<std::uint32_t> w;
              for (int k : idx) w.push_back(c.w[static_cast<std::size_t>(k)]);
              p.v = constant(rt, w);
              return p;
            }
            Val r = tempOf(rt);
            for (std::size_t k = 0; k < idx.size(); ++k) emit(MOV, r.slot + static_cast<std::uint32_t>(k), src.slot + static_cast<std::uint32_t>(idx[k]));
            p.v = r;
            return p;
          }
          const int k = swizzleIndex(e.s[0]);
          if (k < 0 || k >= t->n) error(e.line, "bad component ." + e.s);
          offset = static_cast<std::uint32_t>(k);
          mt = t->el;
        } else {
          error(e.line, "member access on " + name(t));
        }
        return offsetPlace(base, mt, offset);
      }
      // indexing
      if (t->k != K::Vec && t->k != K::Arr) error(e.line, "cannot index " + name(t));
      const Ty* el = t->el;
      Val idx = expr(*e.b);
      if (idx.t->k != K::I32 && idx.t->k != K::U32 && idx.t->k != K::AInt) error(e.line, "index must be an integer");
      if (idx.isConst) {
        const std::int64_t i = isAbstract(idx.t) ? idx.ai : (idx.t->k == K::I32 ? static_cast<std::int32_t>(idx.w[0]) : idx.w[0]);
        if (i < 0 || (t->n >= 0 && i >= t->n)) error(e.line, "index " + std::to_string(i) + " out of bounds");
        return offsetPlace(base, el, static_cast<std::uint32_t>(i) * el->words);
      }
      Ref r;
      if (base.isRef) {
        r = base.r;
      } else {
        r.sp = Ref::Mem;
        r.base = slotOf(base.v);
        r.writable = false;
        r.priv = base.v.priv;
      }
      std::uint32_t off = idx.slot;
      if (t->n >= 0) {
        const std::uint32_t c = alloc(1);
        emit(UMIN, c, off, constU32(static_cast<std::uint32_t>(t->n - 1)));
        off = c;
      }
      if (el->words != 1) {
        const std::uint32_t c = alloc(1);
        emit(IMUL, c, off, constU32(el->words));
        off = c;
      }
      if (r.dyn >= 0) {
        const std::uint32_t c = alloc(1);
        emit(IADD, c, static_cast<std::uint32_t>(r.dyn), off);
        off = c;
      }
      r.dyn = off;
      r.t = el;
      p.isRef = true;
      p.r = r;
      return p;
    }
    p.v = expr(e);
    return p;
  }

  Place offsetPlace(const Place& base, const Ty* t, std::uint32_t offset) {
    Place p;
    if (base.isRef) {
      p.isRef = true;
      p.r = base.r;
      p.r.base += offset;
      p.r.t = t;
      return p;
    }
    if (base.v.isConst) {
      const Val c = concretize(base.v);
      p.v = constant(t, std::vector<std::uint32_t>(c.w.begin() + offset, c.w.begin() + offset + t->words));
      return p;
    }
    p.v = base.v;
    p.v.t = t;
    p.v.slot += offset;
    p.v.temp = false;
    return p;
  }

  Val rvalue(const Place& p) { return p.isRef ? load(p.r) : p.v; }

  Ref lref(const Expr& e) {
    Place p = place(e);
    if (!p.isRef) error(e.line, "cannot assign to this expression");
    return p.r;
  }

  // ---------------------------------------------------------------- expressions

  Val expr(const Expr& e, std::int64_t hint = -1) {
    switch (e.k) {
      case Ek::Int:
        if (e.suffix == 'u') return constant(tU32(), {static_cast<std::uint32_t>(e.iv)});
        if (e.suffix == 'i') return constant(tI32(), {static_cast<std::uint32_t>(e.iv)});
        return abstractInt(static_cast<std::int64_t>(e.iv));
      case Ek::Float:
        if (e.suffix == 'f') return constant(tF32(), {W(static_cast<float>(e.fv))});
        return abstractFloat(e.fv);
      case Ek::Bool: return constant(tBool(), {e.bv ? 1u : 0u});
      case Ek::Ident:
      case Ek::Member:
      case Ek::Index: return rvalue(place(e));
      case Ek::Unary: return unaryExpr(e);
      case Ek::Binary: return binaryExpr(e);
      case Ek::Call: return call(e, hint);
      case Ek::AddrOf: error(e.line, "pointers are only supported as arguments of atomic builtins");
    }
    error(e.line, "bad expression");
  }

  Val unaryExpr(const Expr& e) {
    Val a = expr(*e.a);
    const Ty* el = elemOf(a.t);
    if (e.s == "-") {
      if (a.isConst && el->k == K::AInt) return abstractInt(-a.ai);
      if (a.isConst && el->k == K::AFloat) return abstractFloat(-a.af);
      if (el->k == K::F32) return componentwise(FNEG, a.t, {a});
      if (el->k == K::I32) return componentwise(INEG, a.t, {a});
      error(e.line, "cannot negate " + name(a.t));
    }
    if (e.s == "!") {
      if (el->k != K::Bool) error(e.line, "'!' needs a bool");
      return componentwise(LNOT, a.t, {a});
    }
    if (e.s == "~") {
      if (a.isConst && el->k == K::AInt) return abstractInt(~a.ai);
      if (el->k != K::I32 && el->k != K::U32) error(e.line, "'~' needs an integer");
      return componentwise(NOT, a.t, {a});
    }
    error(e.line, "bad unary operator");
  }

  Val logical(const Expr& e) {
    const bool isAnd = e.s == "&&";
    Val a = expr(*e.a);
    if (a.t->k != K::Bool) error(e.line, "'" + e.s + "' needs bool operands");
    if (a.isConst) {
      if ((a.w[0] != 0u) != isAnd) return a;  // false && x, true || x
      Val b = expr(*e.b);
      if (b.t->k != K::Bool) error(e.line, "'" + e.s + "' needs bool operands");
      return b;
    }
    Val r = tempOf(tBool());
    emit(MOV, r.slot, a.slot);
    const std::size_t j = emit(isAnd ? JZ : JNZ, r.slot, 0);
    const std::uint32_t mark = next_;
    Val b = expr(*e.b);
    if (b.t->k != K::Bool) error(e.line, "'" + e.s + "' needs bool operands");
    emit(MOV, r.slot, slotOf(b));
    next_ = std::max(mark, r.slot + 1);
    patch(j, here());
    return r;
  }

  Val binaryExpr(const Expr& e) {
    if (e.s == "&&" || e.s == "||") return logical(e);
    Val a = expr(*e.a);
    Val b = expr(*e.b);
    return binop(e.s, a, b, e.line);
  }

  Val binop(const std::string& op, Val a, Val b, int line) {
    const bool shiftOp = op == "<<" || op == ">>";
    if (shiftOp) {
      if (isAbstract(b.t)) b = convert(b, tU32(), line);
    } else {
      unify(a, b, line);
    }
    const Ty* ea = elemOf(a.t);
    const Ty* eb = elemOf(b.t);
    // both abstract: fold in the abstract domain
    if (a.isConst && b.isConst && isAbstract(ea) && isAbstract(eb) && isScalar(a.t) && isScalar(b.t)) {
      const bool f = ea->k == K::AFloat || eb->k == K::AFloat;
      const double x = a.af, y = b.af;
      const std::int64_t i = a.ai, j = b.ai;
      auto boolean = [&](bool v) { return constant(tBool(), {v ? 1u : 0u}); };
      if (op == "+") return f ? abstractFloat(x + y) : abstractInt(i + j);
      if (op == "-") return f ? abstractFloat(x - y) : abstractInt(i - j);
      if (op == "*") return f ? abstractFloat(x * y) : abstractInt(i * j);
      if (op == "/") {
        if (!f && j == 0) error(line, "division by zero");
        return f ? abstractFloat(x / y) : abstractInt(i / j);
      }
      if (op == "%") return f ? abstractFloat(std::fmod(x, y)) : abstractInt(j == 0 ? 0 : i % j);
      if (op == "<") return boolean(f ? x < y : i < j);
      if (op == "<=") return boolean(f ? x <= y : i <= j);
      if (op == ">") return boolean(f ? x > y : i > j);
      if (op == ">=") return boolean(f ? x >= y : i >= j);
      if (op == "==") return boolean(f ? x == y : i == j);
      if (op == "!=") return boolean(f ? x != y : i != j);
      if (!f && op == "&") return abstractInt(i & j);
      if (!f && op == "|") return abstractInt(i | j);
      if (!f && op == "^") return abstractInt(i ^ j);
      if (!f && op == "<<") return abstractInt(i << (j & 63));
      if (!f && op == ">>") return abstractInt(i >> (j & 63));
      error(line, "bad operands for '" + op + "'");
    }
    if (isAbstract(ea)) a = concretize(a);
    if (isAbstract(eb)) b = concretize(b);
    ea = elemOf(a.t);
    eb = elemOf(b.t);
    if (shiftOp) {
      if ((ea->k != K::I32 && ea->k != K::U32) || eb->k != K::U32) error(line, "shift needs an integer and a u32");
      return componentwise(op == "<<" ? SHL : (ea->k == K::I32 ? SAR : SHR), a.t, {a, b});
    }
    if (ea != eb) error(line, "'" + op + "' on " + name(a.t) + " and " + name(b.t));
    if (a.t->k == K::Vec && b.t->k == K::Vec && a.t->n != b.t->n) error(line, "vector sizes differ");
    const Ty* shape = a.t->k == K::Vec ? a.t : b.t;
    const K k = ea->k;
    auto cmpShape = [&]() { return shape->k == K::Vec ? vec(shape->n, tBool()) : tBool(); };
    auto pick = [&](std::uint32_t f, std::uint32_t i, std::uint32_t u) -> std::uint32_t {
      if (k == K::F32) return f;
      if (k == K::I32) return i;
      if (k == K::U32) return u;
      error(line, "'" + op + "' is not defined for " + name(ea));
    };
    if (op == "+") return componentwise(pick(FADD, IADD, IADD), shape, {a, b});
    if (op == "-") return componentwise(pick(FSUB, ISUB, ISUB), shape, {a, b});
    if (op == "*") return componentwise(pick(FMUL, IMUL, IMUL), shape, {a, b});
    if (op == "/") return componentwise(pick(FDIV, IDIV, UDIV), shape, {a, b});
    if (op == "%") return componentwise(pick(FMOD, IMOD, UMOD), shape, {a, b});
    if (op == "<") return componentwise(pick(FLT, ILT, ULT), cmpShape(), {a, b});
    if (op == "<=") return componentwise(pick(FLE, ILE, ULE), cmpShape(), {a, b});
    if (op == ">") return componentwise(pick(FGT, IGT, UGT), cmpShape(), {a, b});
    if (op == ">=") return componentwise(pick(FGE, IGE, UGE), cmpShape(), {a, b});
    if (op == "==") return componentwise(k == K::F32 ? FEQ : EQ, cmpShape(), {a, b});
    if (op == "!=") return componentwise(k == K::F32 ? FNE : NE, cmpShape(), {a, b});
    if (op == "&" || op == "|" || op == "^") {
      if (k != K::I32 && k != K::U32 && k != K::Bool) error(line, "'" + op + "' needs integers or bools");
      return componentwise(op == "&" ? AND : (op == "|" ? OR : XOR), shape, {a, b});
    }
    error(line, "unknown operator '" + op + "'");
  }

  // ---------------------------------------------------------------- calls

  Val call(const Expr& e, std::int64_t hint) {
    const std::string& n = e.callee.name;
    if (e.templated || isTypeName(n)) return construct(e);
    auto f = fns_.find(n);
    if (f != fns_.end()) return inlineCall(*f->second, e, hint);
    return builtin(e);
  }

  std::vector<std::uint32_t> wordsOf(const Val& v) { return concretize(v).w; }

  // A value made of parts (all constant -> a constant).
  Val compose(const Ty* t, const std::vector<Val>& parts) {
    bool allConst = true;
    for (const auto& p : parts) allConst = allConst && p.isConst;
    if (allConst) {
      std::vector<std::uint32_t> w;
      for (const auto& p : parts) {
        const auto pw = wordsOf(p);
        w.insert(w.end(), pw.begin(), pw.end());
      }
      return constant(t, w);
    }
    Val r = tempOf(t);
    std::uint32_t off = 0;
    for (const auto& p : parts) {
      const std::uint32_t n = p.t->words;
      if (n == 1) emit(MOV, r.slot + off, slotOf(p));
      else emit(MOVN, r.slot + off, slotOf(p), n);
      off += n;
    }
    return r;
  }

  // Value conversion of a scalar to a concrete scalar type.
  Val convertScalar(const Val& v, const Ty* to, int line) {
    if (v.t == to) return v;
    if (isAbstract(v.t)) {
      if (to->k == K::Bool) return constant(tBool(), {(v.t->k == K::AInt ? v.ai != 0 : v.af != 0.0) ? 1u : 0u});
      if (v.t->k == K::AFloat && (to->k == K::I32 || to->k == K::U32))
        return componentwise(to->k == K::I32 ? F2I : F2U, to, {constant(tF32(), {W(static_cast<float>(v.af))})});
      return convert(v, to, line);
    }
    const K from = v.t->k;
    auto relabel = [&](const Val& x) {
      if (x.isConst) return constant(to, x.w);
      Val r = x;
      r.t = to;
      r.temp = false;
      return r;
    };
    switch (to->k) {
      case K::F32:
        if (from == K::I32) return componentwise(I2F, to, {v});
        if (from == K::U32) return componentwise(U2F, to, {v});
        if (from == K::Bool) return componentwise(B2F, to, {v});
        break;
      case K::I32:
        if (from == K::F32) return componentwise(F2I, to, {v});
        if (from == K::U32 || from == K::Bool) return relabel(v);
        break;
      case K::U32:
        if (from == K::F32) return componentwise(F2U, to, {v});
        if (from == K::I32 || from == K::Bool) return relabel(v);
        break;
      case K::Bool:
        if (from == K::F32) return componentwise(F2B, to, {v});
        if (from == K::I32 || from == K::U32) return componentwise(I2B, to, {v});
        break;
      default: break;
    }
    error(line, "cannot convert " + name(v.t) + " to " + name(to));
  }

  // The scalar components of a value, each as a Val.
  std::vector<Val> components(const Val& v) {
    std::vector<Val> out;
    if (v.t->k != K::Vec) {
      out.push_back(v);
      return out;
    }
    for (int i = 0; i < v.t->n; ++i) {
      if (v.isConst) {
        out.push_back(constant(v.t->el, {v.w[static_cast<std::size_t>(i)]}));
      } else {
        Val c = v;
        c.t = v.t->el;
        c.slot = v.slot + static_cast<std::uint32_t>(i);
        c.temp = false;
        out.push_back(c);
      }
    }
    return out;
  }

  Val construct(const Expr& e) {
    const std::string& n = e.callee.name;
    std::vector<Val> args;
    for (const auto& a : e.args) args.push_back(expr(*a));
    const int line = e.line;
    if (n == "f32" || n == "u32" || n == "i32" || n == "bool") {
      const Ty* t = resolve(e.callee);
      if (args.empty()) return constant(t, {0u});
      if (args.size() != 1 || !isScalar(args[0].t)) error(line, n + "() takes one scalar");
      return convertScalar(args[0], t, line);
    }
    if (structIds_.count(n)) {
      const Ty* t = resolve(e.callee);
      const auto& members = structs_[static_cast<std::size_t>(t->sid)].members;
      if (args.empty()) return constant(t, std::vector<std::uint32_t>(t->words, 0u));
      if (args.size() != members.size()) error(line, n + " takes " + std::to_string(members.size()) + " values");
      std::vector<Val> parts;
      for (std::size_t i = 0; i < args.size(); ++i) parts.push_back(convert(args[i], members[i].t, line));
      return compose(t, parts);
    }
    if (n == "array") {
      const Ty* el = nullptr;
      if (!e.callee.args.empty()) el = resolve(e.callee.args[0]);
      if (!el) {
        for (const auto& a : args)
          if (!isAbstract(a.t)) el = a.t;
        if (!el) el = concretize(args.at(0)).t;
      }
      const int count = e.callee.count ? resolve(e.callee)->n : static_cast<int>(args.size());
      const Ty* t = arr(el, count);
      if (args.empty()) return constant(t, std::vector<std::uint32_t>(t->words, 0u));
      if (static_cast<int>(args.size()) != count) error(line, "array constructor needs " + std::to_string(count) + " values");
      std::vector<Val> parts;
      for (auto& a : args) parts.push_back(convert(a, el, line));
      return compose(t, parts);
    }
    // vectors
    int size = n[3] - '0';
    const Ty* el = nullptr;
    if (!e.callee.args.empty()) el = resolve(e.callee.args[0]);
    else if (n.size() == 5) el = n[4] == 'f' ? tF32() : (n[4] == 'u' ? tU32() : tI32());
    std::vector<Val> scalars;
    for (const auto& a : args)
      for (const auto& c : components(a)) scalars.push_back(c);
    if (!el) {
      for (const auto& s : scalars)
        if (!isAbstract(s.t)) el = s.t;
      if (!el) el = scalars.empty() ? tF32() : concretize(scalars[0]).t;
    }
    const Ty* t = vec(size, el);
    if (scalars.empty()) return constant(t, std::vector<std::uint32_t>(t->words, 0u));
    if (scalars.size() == 1) scalars.assign(static_cast<std::size_t>(size), scalars[0]);
    if (static_cast<int>(scalars.size()) != size) error(line, "vector constructor needs " + std::to_string(size) + " components");
    std::vector<Val> parts;
    for (auto& s : scalars) parts.push_back(convertScalar(s, el, line));
    return compose(t, parts);
  }

  Val inlineCall(const FnDecl& f, const Expr& e, std::int64_t hint) {
    if (++depth_ > 64) error(e.line, "calls nested too deeply (recursion?)");
    if (e.args.size() != f.params.size()) error(e.line, f.name + " takes " + std::to_string(f.params.size()) + " arguments");
    std::vector<Val> args;
    for (std::size_t i = 0; i < e.args.size(); ++i) args.push_back(convert(expr(*e.args[i]), resolve(f.params[i].type), e.line));
    const Ty* rt = f.hasRet ? resolve(f.ret) : scalar(K::Void);
    Val result;
    result.t = rt;
    if (f.hasRet) {
      if (hint >= 0) {
        result.slot = static_cast<std::uint32_t>(hint);
      } else {
        result.slot = alloc(rt->words);
        result.temp = true;
        result.immutable = true;
      }
    }
    const std::uint32_t mark = next_;
    auto savedScopes = std::move(scopes_);
    auto savedLoops = std::move(loops_);
    scopes_.assign(1, {});
    loops_.clear();
    for (std::size_t i = 0; i < args.size(); ++i) {
      Binding b;
      b.kind = Binding::Let;
      const Val& v = args[i];
      if (v.isConst || ((v.temp || v.immutable) && !v.priv)) {
        b.v = v;
        b.v.temp = false;
        b.v.immutable = true;
      } else {
        Val c = tempOf(v.t);
        if (v.t->words == 1) emit(MOV, c.slot, slotOf(v));
        else emit(MOVN, c.slot, slotOf(v), v.t->words);
        b.v = c;
        b.v.temp = false;
      }
      scopes_.back()[f.params[i].name] = b;
    }
    rets_.push_back({rt, result.slot, {}});
    stmt(*f.body);
    const std::uint32_t end = here();
    for (std::size_t j : rets_.back().jumps) patch(j, end);
    rets_.pop_back();
    scopes_ = std::move(savedScopes);
    loops_ = std::move(savedLoops);
    next_ = mark;
    --depth_;
    return result;
  }

  Val math1(std::uint32_t op, const Val& x, int line) {
    Val v = isAbstract(elemOf(x.t)) ? convert(x, x.t->k == K::Vec ? vec(x.t->n, tF32()) : tF32(), line) : x;
    if (elemOf(v.t)->k != K::F32) error(line, "expects f32");
    return componentwise(op, v.t, {v});
  }

  Val dotOf(Val a, Val b, int line) {
    unify(a, b, line);
    if (a.t->k != K::Vec || a.t != b.t) error(line, "dot needs two vectors of the same type");
    const K k = a.t->el->k;
    const std::uint32_t mul = k == K::F32 ? FMUL : IMUL, add = k == K::F32 ? FADD : IADD;
    const auto ca = components(a), cb = components(b);
    Val acc = componentwise(mul, a.t->el, {ca[0], cb[0]});
    for (std::size_t i = 1; i < ca.size(); ++i) acc = componentwise(add, a.t->el, {acc, componentwise(mul, a.t->el, {ca[i], cb[i]})});
    return acc;
  }

  Ref atomicRef(const Expr& e) {
    if (e.k != Ek::AddrOf) error(e.line, "atomic builtins take a pointer (&x)");
    Ref r = lref(*e.a);
    if (r.t->k != K::Atomic || r.sp != Ref::Wg) error(e.line, "only workgroup atomics are supported");
    return r;
  }

  Val builtin(const Expr& e) {
    const std::string& n = e.callee.name;
    const int line = e.line;
    if (n == "workgroupBarrier" || n == "storageBarrier") {
      emit(BARRIER);
      out_.hasBarriers = true;
      return voidVal();
    }
    if (n == "atomicLoad" || n == "atomicStore" || n == "atomicAdd" || n == "atomicSub") {
      Ref r = atomicRef(*e.args.at(0));
      const Ty* el = r.t->el;
      const std::uint32_t off = r.dyn >= 0 ? static_cast<std::uint32_t>(r.dyn) : constU32(0);
      if (n == "atomicLoad") {
        Val t = tempOf(el);
        emit(ATLD, t.slot, r.base, off);
        return t;
      }
      Val v = convert(expr(*e.args.at(1)), el, line);
      if (n == "atomicStore") {
        emit(ATST, r.base, off, slotOf(v));
        return voidVal();
      }
      if (n == "atomicSub") v = componentwise(INEG, el, {v});
      Val t = tempOf(el);
      emit(ATADD, t.slot, r.base, off, slotOf(v));
      return t;
    }
    std::vector<Val> a;
    for (const auto& x : e.args) a.push_back(expr(*x));
    auto need = [&](std::size_t k) {
      if (a.size() != k) error(line, n + " takes " + std::to_string(k) + " arguments");
    };
    static const std::unordered_map<std::string, std::uint32_t> unaryF = {
        {"sqrt", FSQRT}, {"inverseSqrt", FINVSQRT}, {"exp", FEXP}, {"exp2", FEXP2}, {"log", FLOG},   {"log2", FLOG2},
        {"sin", FSIN},   {"cos", FCOS},             {"tan", FTAN}, {"asin", FASIN}, {"acos", FACOS}, {"atan", FATAN},
        {"sinh", FSINH}, {"cosh", FCOSH},           {"tanh", FTANH}, {"floor", FFLOOR}, {"ceil", FCEIL}, {"round", FROUND},
        {"trunc", FTRUNC}, {"fract", FFRACT},       {"sign", FSIGN}};
    auto u = unaryF.find(n);
    if (u != unaryF.end()) {
      need(1);
      return math1(u->second, a[0], line);
    }
    if (n == "abs") {
      need(1);
      Val x = isAbstract(elemOf(a[0].t)) ? concretize(a[0]) : a[0];
      const K k = elemOf(x.t)->k;
      if (k == K::U32) return x;
      return componentwise(k == K::F32 ? FABS : IABS, x.t, {x});
    }
    if (n == "pow" || n == "atan2" || n == "min" || n == "max" || n == "step") {
      need(2);
      Val x = a[0], y = a[1];
      unify(x, y, line);
      if (isAbstract(elemOf(x.t))) x = concretize(x);
      if (isAbstract(elemOf(y.t))) y = concretize(y);
      const K k = elemOf(x.t)->k;
      const Ty* shape = x.t->k == K::Vec ? x.t : y.t;
      if (n == "pow") return componentwise(FPOW, shape, {x, y});
      if (n == "atan2") return componentwise(FATAN2, shape, {x, y});
      if (n == "step") return componentwise(B2F, shape, {componentwise(FGE, shape->k == K::Vec ? vec(shape->n, tBool()) : tBool(), {y, x})});
      if (n == "min") return componentwise(k == K::F32 ? FMIN : (k == K::I32 ? IMIN : UMIN), shape, {x, y});
      return componentwise(k == K::F32 ? FMAX : (k == K::I32 ? IMAX : UMAX), shape, {x, y});
    }
    if (n == "clamp") {
      need(3);
      Val x = a[0], lo = a[1], hi = a[2];
      unify(x, lo, line);
      unify(x, hi, line);
      unify(lo, hi, line);
      if (isAbstract(elemOf(x.t))) {
        x = concretize(x);
        unify(x, lo, line);
        unify(x, hi, line);
      }
      const K k = elemOf(x.t)->k;
      const std::uint32_t mx = k == K::F32 ? FMAX : (k == K::I32 ? IMAX : UMAX), mn = k == K::F32 ? FMIN : (k == K::I32 ? IMIN : UMIN);
      return componentwise(mn, x.t, {componentwise(mx, x.t, {x, lo}), hi});
    }
    if (n == "select") {
      need(3);
      Val f = a[0], t = a[1], c = a[2];
      unify(f, t, line);
      if (isAbstract(elemOf(f.t))) {
        f = concretize(f);
        t = concretize(t);
      }
      if (elemOf(c.t)->k != K::Bool) error(line, "select needs a bool condition");
      const Ty* shape = f.t->k == K::Vec ? f.t : t.t;
      if (f.t->words != shape->words || t.t->words != shape->words) error(line, "select needs values of the same type");
      if (c.isConst && c.t->k == K::Bool) return c.w[0] != 0u ? t : f;
      if (shape->k != K::Vec && !isScalar(shape)) {  // structs and arrays: copy by condition
        Val r = tempOf(shape);
        emit(MOVN, r.slot, slotOf(f), shape->words);
        const std::size_t j = emit(JZ, slotOf(c), 0);
        emit(MOVN, r.slot, slotOf(t), shape->words);
        patch(j, here());
        return r;
      }
      return componentwise(SEL, shape, {f, t, c});
    }
    if (n == "mix") {
      need(3);
      Val x = a[0], y = a[1], t = a[2];
      unify(x, y, line);
      unify(x, t, line);
      if (isAbstract(elemOf(x.t))) {
        x = concretize(x);
        unify(x, y, line);
        unify(x, t, line);
      }
      const Ty* shape = x.t;
      Val one = constant(tF32(), {W(1.0f)});
      Val oneMinus = componentwise(FSUB, t.t, {one, t});
      return componentwise(FADD, shape, {componentwise(FMUL, shape, {x, oneMinus}), componentwise(FMUL, shape, {y, t})});
    }
    if (n == "dot") {
      need(2);
      return dotOf(a[0], a[1], line);
    }
    if (n == "length") {
      need(1);
      if (a[0].t->k != K::Vec) return math1(FABS, a[0], line);
      return math1(FSQRT, dotOf(a[0], a[0], line), line);
    }
    if (n == "distance") {
      need(2);
      Val d = binop("-", a[0], a[1], line);
      if (d.t->k != K::Vec) return math1(FABS, d, line);
      return math1(FSQRT, dotOf(d, d, line), line);
    }
    if (n == "normalize") {
      need(1);
      if (a[0].t->k != K::Vec) error(line, "normalize needs a vector");
      Val len = math1(FSQRT, dotOf(a[0], a[0], line), line);
      return componentwise(FDIV, a[0].t, {a[0], len});
    }
    error(line, "unknown function '" + n + "'");
  }

  // ---------------------------------------------------------------- statements

  void bindValue(const std::string& nm, Binding b, int line) {
    auto& scope = scopes_.back();
    if (scope.count(nm)) error(line, "'" + nm + "' is already declared in this scope");
    scope[nm] = std::move(b);
  }

  void stmt(const Stmt& s) {
    const std::uint32_t mark = next_;
    switch (s.k) {
      case Sk::Empty: return;
      case Sk::Block: {
        scopes_.emplace_back();
        for (const auto& x : s.body) stmt(*x);
        scopes_.pop_back();
        next_ = mark;
        return;
      }
      case Sk::Let:
      case Sk::Const: {
        Val v = expr(*s.init);
        if (s.hasType) v = convert(v, resolve(s.type), s.line);
        else if (v.isConst) v = concretize(v);
        if (s.k == Sk::Const && !v.isConst) error(s.line, "const '" + s.name + "' is not constant");
        Binding b;
        b.kind = v.isConst ? Binding::Const : Binding::Let;
        if (v.isConst || v.temp) {
          b.v = v;
          if (v.temp) next_ = std::max(mark, v.slot + v.t->words);
          else next_ = mark;
        } else if (v.immutable && !v.priv) {
          b.v = v;
          next_ = mark;
        } else {
          next_ = mark;
          Val c = tempOf(v.t);
          if (v.t->words == 1) emit(MOV, c.slot, slotOf(v));
          else emit(MOVN, c.slot, slotOf(v), v.t->words);
          b.v = c;
        }
        b.v.temp = false;
        b.v.immutable = true;
        b.v.priv = false;
        bindValue(s.name, b, s.line);
        return;
      }
      case Sk::Var: {
        const Ty* t = s.hasType ? resolve(s.type) : nullptr;
        Val v;
        bool has = false;
        if (s.init) {
          v = expr(*s.init);
          if (t) v = convert(v, t, s.line);
          else v = concretize(v);
          t = v.t;
          has = true;
        }
        if (!t) error(s.line, "var '" + s.name + "' needs a type or an initializer");
        Binding b;
        b.kind = Binding::Var;
        b.r.sp = Ref::Mem;
        b.r.t = t;
        if (has && !v.isConst && v.temp) {
          b.r.base = v.slot;
          next_ = std::max(mark, v.slot + t->words);
        } else {
          next_ = mark;
          b.r.base = alloc(t->words);
          const std::uint32_t src = has ? slotOf(v) : zeroWords(t->words);
          if (t->words == 1) emit(MOV, b.r.base, src);
          else emit(MOVN, b.r.base, src, t->words);
        }
        bindValue(s.name, b, s.line);
        return;
      }
      case Sk::Assign: {
        Ref r = lref(*s.lhs);
        if (s.op == "=") {
          if (r.sp == Ref::Mem && r.dyn < 0 && s.init->k == Ek::Call && fns_.count(s.init->callee.name) && r.writable) {
            Val v = expr(*s.init, static_cast<std::int64_t>(r.base));
            if (v.t != r.t) error(s.line, "cannot assign " + name(v.t) + " to " + name(r.t));
          } else {
            store(r, expr(*s.init), s.line);
          }
        } else {
          const std::string op = s.op.substr(0, s.op.size() - 1);
          Val cur = load(r);
          cur.temp = false;
          store(r, binop(op, cur, expr(*s.init), s.line), s.line);
        }
        next_ = mark;
        return;
      }
      case Sk::IncDec: {
        Ref r = lref(*s.lhs);
        Val cur = load(r);
        cur.temp = false;
        store(r, binop(s.op == "++" ? "+" : "-", cur, abstractInt(1), s.line), s.line);
        next_ = mark;
        return;
      }
      case Sk::Phony:
      case Sk::Call: {
        expr(*s.init);
        next_ = mark;
        return;
      }
      case Sk::If: {
        Val c = expr(*s.cond);
        if (c.t->k != K::Bool) error(s.line, "if needs a bool condition");
        next_ = mark;
        if (c.isConst) {
          if (c.w[0] != 0u) scoped(*s.then);
          else if (s.els) scoped(*s.els);
          return;
        }
        const std::size_t jz = emit(JZ, c.slot, 0);
        scoped(*s.then);
        if (s.els) {
          const std::size_t jmp = emit(JMP, 0);
          patch(jz, here());
          scoped(*s.els);
          patch(jmp, here());
        } else {
          patch(jz, here());
        }
        return;
      }
      case Sk::For:
      case Sk::While:
      case Sk::Loop: {
        if (s.k == Sk::For && unrolled(s)) {
          next_ = mark;
          return;
        }
        scopes_.emplace_back();
        if (s.first) stmt(*s.first);
        const std::uint32_t loopMark = next_;
        const std::uint32_t top = here();
        loops_.emplace_back();
        std::size_t exitJump = static_cast<std::size_t>(-1);
        if (s.cond) {
          Val c = expr(*s.cond);
          if (c.t->k != K::Bool) error(s.line, "loop condition must be a bool");
          if (!c.isConst) exitJump = emit(JZ, c.slot, 0);
          else if (c.w[0] == 0u) exitJump = emit(JMP, 0);
          next_ = loopMark;
        }
        scoped(*s.then);
        const std::uint32_t cont = here();
        if (s.update) stmt(*s.update);
        next_ = loopMark;
        emit(JMP, top);
        const std::uint32_t end = here();
        if (exitJump != static_cast<std::size_t>(-1)) patch(exitJump, end);
        for (std::size_t j : loops_.back().breaks) patch(j, end);
        for (std::size_t j : loops_.back().continues) patch(j, cont);
        loops_.pop_back();
        scopes_.pop_back();
        next_ = mark;
        return;
      }
      case Sk::Break:
        if (loops_.empty()) error(s.line, "break outside a loop");
        loops_.back().breaks.push_back(emit(JMP, 0));
        return;
      case Sk::Continue:
        if (loops_.empty()) error(s.line, "continue outside a loop");
        loops_.back().continues.push_back(emit(JMP, 0));
        return;
      case Sk::Return: {
        if (rets_.empty()) {
          emit(END);
          return;
        }
        const Ty* rt = rets_.back().t;  // (expr may inline calls, which grow rets_)
        const std::uint32_t rslot = rets_.back().slot;
        if (s.init) {
          if (rt->k == K::Void) error(s.line, "this function returns nothing");
          const bool direct = s.init->k == Ek::Call && fns_.count(s.init->callee.name) > 0;
          Val v = convert(expr(*s.init, direct ? static_cast<std::int64_t>(rslot) : -1), rt, s.line);
          const std::uint32_t n = rt->words;
          if (!(!v.isConst && v.slot == rslot)) {
            if (n == 1 && !v.isConst && v.temp && lastWrites(v.slot)) code_.back().a = rslot;
            else if (n == 1) emit(MOV, rslot, slotOf(v));
            else emit(MOVN, rslot, slotOf(v), n);
          }
        } else if (rt->k != K::Void) {
          error(s.line, "missing return value");
        }
        rets_.back().jumps.push_back(emit(JMP, 0));
        next_ = mark;
        return;
      }
    }
  }

  // ---------------------------------------------------------------- loop unrolling
  //
  // for (var i = c0; i < c1; i = i + c2) { body } with constant bounds, a body that never assigns i,
  // breaks or continues this loop, and a small trip count × body is unrolled: every iteration is
  // the body with i bound to a constant, so indices into arrays fold to fixed offsets and the
  // loop's compare, branch and increment disappear.

  // A constant value of an expression, or nothing (any code it emitted is discarded).
  bool constantOf(const Expr& e, Val& out) {
    const std::size_t codeMark = code_.size(), label = lastLabel_;
    const std::uint32_t slotMark = next_;
    Val v = expr(e);
    if (v.isConst) {
      out = v;
      return true;
    }
    code_.resize(codeMark);
    lastLabel_ = label;
    next_ = slotMark;
    return false;
  }

  static bool isIdent(const ExprP& e, const std::string& n) { return e && e->k == Ek::Ident && e->s == n; }

  static std::size_t weight(const Expr& e) {
    std::size_t w = e.k == Ek::Call ? 24 : 1;
    for (const auto& a : e.args) w += weight(*a);
    if (e.a) w += weight(*e.a);
    if (e.b) w += weight(*e.b);
    return w;
  }

  // Whether a statement assigns `n`, or breaks or continues the loop it is in; and its size.
  static bool blocks(const Stmt& s, const std::string& n, bool nested, std::size_t& w) {
    const auto exprW = [&](const ExprP& e) {
      if (e) w += weight(*e);
    };
    exprW(s.init);
    exprW(s.lhs);
    exprW(s.cond);
    ++w;
    if ((s.k == Sk::Assign || s.k == Sk::IncDec) && s.lhs) {
      const Expr* root = s.lhs.get();
      while (root->k == Ek::Member || root->k == Ek::Index) root = root->a.get();
      if (root->k == Ek::Ident && root->s == n) return true;
    }
    if ((s.k == Sk::Var || s.k == Sk::Let || s.k == Sk::Const) && s.name == n) return true;  // shadowing
    if (!nested && (s.k == Sk::Break || s.k == Sk::Continue)) return true;
    const bool inner = nested || s.k == Sk::For || s.k == Sk::While || s.k == Sk::Loop;
    for (const auto& x : s.body)
      if (blocks(*x, n, inner, w)) return true;
    for (const StmtP* x : {&s.then, &s.els, &s.first, &s.update})
      if (*x && blocks(**x, n, inner, w)) return true;
    return false;
  }

  bool unrolled(const Stmt& s) {
    if (!s.first || s.first->k != Sk::Var || !s.first->init || !s.cond || !s.update || !s.then) return false;
    const std::string& n = s.first->name;
    // i < c1, i <= c1, i != c1
    const Expr& c = *s.cond;
    if (c.k != Ek::Binary || !isIdent(c.a, n) || (c.s != "<" && c.s != "<=" && c.s != "!=")) return false;
    // i = i + c2, i += c2, i++
    ExprP stepExpr;
    const Stmt& u = *s.update;
    if (u.k == Sk::IncDec && u.op == "++" && isIdent(u.lhs, n)) {
      stepExpr = nullptr;
    } else if (u.k == Sk::Assign && isIdent(u.lhs, n) && u.op == "+=") {
      stepExpr = u.init;
    } else if (u.k == Sk::Assign && isIdent(u.lhs, n) && u.op == "=" && u.init->k == Ek::Binary && u.init->s == "+" && isIdent(u.init->a, n)) {
      stepExpr = u.init->b;
    } else {
      return false;
    }
    std::size_t w = 0;
    if (blocks(*s.then, n, false, w)) return false;
    Val v0, v1, vs;
    if (!constantOf(*s.first->init, v0) || !constantOf(*c.b, v1)) return false;
    if (stepExpr && !constantOf(*stepExpr, vs)) return false;
    const Ty* t = s.first->hasType ? resolve(s.first->type) : concretize(v0).t;
    if (t->k != K::U32 && t->k != K::I32) return false;
    auto value = [&](const Val& v) -> std::int64_t {
      const Val x = convert(v, t, s.line);
      return t->k == K::U32 ? static_cast<std::int64_t>(x.w[0]) : static_cast<std::int64_t>(static_cast<std::int32_t>(x.w[0]));
    };
    const std::int64_t from = value(v0), to = value(v1), step = stepExpr ? value(vs) : 1;
    if (step <= 0) return false;
    std::vector<std::int64_t> values;
    for (std::int64_t i = from; c.s == "<" ? i < to : (c.s == "<=" ? i <= to : i != to); i += step) {
      values.push_back(i);
      if (values.size() > 128) return false;
    }
    if (values.size() * w > 6000) return false;
    for (std::int64_t i : values) {
      scopes_.emplace_back();
      Binding b;
      b.kind = Binding::Const;
      b.v = constant(t, {static_cast<std::uint32_t>(i)});
      scopes_.back()[n] = b;
      scoped(*s.then);
      scopes_.pop_back();
    }
    return true;
  }

  void scoped(const Stmt& s) {
    const std::uint32_t mark = next_;
    scopes_.emplace_back();
    stmt(s);
    scopes_.pop_back();
    next_ = mark;
  }
};

}  // namespace

std::shared_ptr<const Shader> Shader::compile(const std::string& wgsl, const std::string& entry) {
  auto shader = std::make_shared<Shader>();
  Parser parser(lex(wgsl));
  Compiler compiler(parser.module(), *shader);
  compiler.compile(entry);
  return shader;
}

}  // namespace syn::gpu
