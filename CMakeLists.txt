cmake_minimum_required(VERSION 3.16)

project(synaptic
  VERSION 0.1.0
  DESCRIPTION "Learning theories as computational models: how people learn"
  LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

option(SYN_BUILD_TESTS "Build the unit tests" ON)
option(SYN_BUILD_EXAMPLES "Build the example programs" ON)

# Every model of the library is one file in src/models (listed in src/models/all_models.cpp).
file(GLOB SYN_MODEL_SOURCES CONFIGURE_DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/src/models/*.cpp)

add_library(syn
  src/core/multivector.cpp
  src/core/elliptic_cga.cpp
  src/core/value.cpp
  src/space/space.cpp
  src/space/algebra.cpp
  src/kernel/kernel.cpp
  src/gpu/wgsl_compiler.cpp
  src/gpu/software_gpu.cpp
  ${SYN_MODEL_SOURCES}
)
add_library(syn::syn ALIAS syn)

if(NOT EMSCRIPTEN)
  # The software GPU spreads a dispatch's workgroups over the cores.
  find_package(Threads REQUIRED)
  target_link_libraries(syn PUBLIC Threads::Threads)
endif()

target_include_directories(syn PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>)

if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
  target_compile_options(syn PRIVATE -Wall -Wextra -Wpedantic -Wshadow)
elseif(MSVC)
  target_compile_options(syn PRIVATE /W4)
endif()

if(EMSCRIPTEN)
  # WebAssembly build of the library with the Embind wrapper used by the web front end:
  #   emcmake cmake -S . -B build-wasm && cmake --build build-wasm
  # produces web/wasm/syn.js + syn.wasm.
  set(SYN_WASM_FLAGS -fwasm-exceptions -msimd128)
  target_compile_options(syn PRIVATE ${SYN_WASM_FLAGS})
  add_executable(syn_wasm wasm/bindings.cpp)
  target_compile_options(syn_wasm PRIVATE ${SYN_WASM_FLAGS})
  # The wrapper is glue only. When functions are inlined into the try/catch guards, LLVM places
  # wasm exception-handling try/delegate blocks non-deterministically, so the committed syn.wasm
  # could not be checked against a rebuild. -O2 without inlining is byte-reproducible; the
  # library itself stays at -O3.
  set_source_files_properties(wasm/bindings.cpp PROPERTIES COMPILE_OPTIONS "-O2;-fno-inline-functions")
  target_compile_definitions(syn_wasm PRIVATE SYN_VERSION="${PROJECT_VERSION}")
  target_link_libraries(syn_wasm PRIVATE syn)
  target_link_options(syn_wasm PRIVATE
    -lembind -fwasm-exceptions
    -sMODULARIZE=1 -sEXPORT_ES6=1 -sEXPORT_NAME=createSynModule
    -sALLOW_MEMORY_GROWTH=1 -sENVIRONMENT=web,worker,node)
  set_target_properties(syn_wasm PROPERTIES
    OUTPUT_NAME syn SUFFIX ".js"
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/web/wasm)
  set(SYN_BUILD_TESTS OFF)
  set(SYN_BUILD_EXAMPLES OFF)
endif()

if(SYN_BUILD_TESTS)
  enable_testing()
  add_executable(syn_tests
    tests/test_main.cpp
    tests/test_cga.cpp
    tests/test_space.cpp
    tests/test_software_gpu.cpp
    tests/test_models.cpp
  )
  target_link_libraries(syn_tests PRIVATE syn)
  add_test(NAME syn_tests COMMAND syn_tests)
endif()

if(SYN_BUILD_EXAMPLES)
  add_executable(learning_theories_demo examples/learning_theories_demo.cpp)
  target_link_libraries(learning_theories_demo PRIVATE syn)
  # The end-to-end QA golden test (tools/qa_golden/README.md); its source is not in every build context.
  if(NOT EMSCRIPTEN AND EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/tools/qa_golden/qa_golden.cpp)
    add_executable(qa_golden tools/qa_golden/qa_golden.cpp)
    target_link_libraries(qa_golden PRIVATE syn)
  endif()
  # The reader as a native tool (tools/reader/reader.hpp): asked from a shell or a server (svr-net/ui).
  if(NOT EMSCRIPTEN AND EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/tools/reader/reader.cpp)
    add_executable(synaptic_reader tools/reader/main.cpp tools/reader/reader.cpp)
    target_link_libraries(synaptic_reader PRIVATE syn)
    if(SYN_BUILD_TESTS)
      add_test(NAME reader_answers_from_what_it_knows COMMAND synaptic_reader ask "What does the Michigan Makers program help students do?")
      set_tests_properties(reader_answers_from_what_it_knows PROPERTIES PASS_REGULAR_EXPRESSION "\"found\":true.*Michigan Makers")
      add_test(NAME reader_answers_from_passages_given COMMAND sh -c "printf 'A new lotion speeds DNA repair in mice.\\nDNA is the molecule that carries the genetic instructions of living things.\\n' | $<TARGET_FILE:synaptic_reader> from What is DNA?")
      set_tests_properties(reader_answers_from_passages_given PROPERTIES PASS_REGULAR_EXPRESSION "molecule.*\"confident\":true.*\"passages\":2")
      add_test(NAME reader_reports_what_it_knows COMMAND synaptic_reader info)
      set_tests_properties(reader_reports_what_it_knows PROPERTIES PASS_REGULAR_EXPRESSION "\"passages\":60")
    endif()
  endif()
endif()

include(GNUInstallDirs)
install(TARGETS syn EXPORT synTargets
  ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR})
install(DIRECTORY include/syn DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
install(EXPORT synTargets NAMESPACE syn:: DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/syn)
