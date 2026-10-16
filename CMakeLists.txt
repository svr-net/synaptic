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

add_library(syn
  src/core/session_grid.cpp
  src/behaviorism/classical_conditioning.cpp
  src/behaviorism/operant_conditioning.cpp
  src/behaviorism/token_economy.cpp
  src/humanism/maslow.cpp
  src/humanism/information_search.cpp
  src/cognitivism/memory.cpp
  src/cognitivism/interleaving.cpp
  src/cognitivism/schema.cpp
  src/constructivism/equilibration.cpp
  src/social/zpd.cpp
  src/social/dialogue.cpp
  src/development/piaget.cpp
  src/development/perry.cpp
  src/andragogy/andragogy.cpp
  src/motivation/expectancy_value.cpp
  src/motivation/mindset.cpp
  src/classroom/learner.cpp
  src/classroom/instructional_design.cpp
  src/classroom/classroom_engine.cpp
  src/classroom/allocation.cpp
  src/classroom/sensitivities.cpp
)
add_library(syn::syn ALIAS syn)

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
    tests/test_core.cpp
    tests/test_behaviorism.cpp
    tests/test_humanism.cpp
    tests/test_cognitivism.cpp
    tests/test_constructivism.cpp
    tests/test_social.cpp
    tests/test_development.cpp
    tests/test_andragogy.cpp
    tests/test_motivation.cpp
    tests/test_classroom.cpp
  )
  target_link_libraries(syn_tests PRIVATE syn)
  add_test(NAME syn_tests COMMAND syn_tests)
endif()

if(SYN_BUILD_EXAMPLES)
  add_executable(learning_theories_demo examples/learning_theories_demo.cpp)
  target_link_libraries(learning_theories_demo PRIVATE syn)
endif()

include(GNUInstallDirs)
install(TARGETS syn EXPORT synTargets
  ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR})
install(DIRECTORY include/syn DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
install(EXPORT synTargets NAMESPACE syn:: DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/syn)
