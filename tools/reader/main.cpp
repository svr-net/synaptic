// synaptic_reader: the reader as a native tool (reader.hpp).
#include <iostream>

#include "reader.hpp"

int main(int argc, char** argv) {
  std::ios::sync_with_stdio(false);
  return syn::reader::run(std::vector<std::string>(argv + 1, argv + argc), {}, std::cin, std::cout, std::cerr);
}
