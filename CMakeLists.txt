cmake_minimum_required(VERSION 3.20)
project(chess-ai VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

add_library(chess_core
    src/core/bitboard.cpp
    src/core/attacks.cpp
    src/core/position.cpp
    src/core/movegen.cpp
    src/mcts/node.cpp
    src/mcts/search.cpp
    src/neural/policy_map.cpp
)
target_include_directories(chess_core PUBLIC src)

add_executable(chess_engine src/main.cpp)
target_link_libraries(chess_engine PRIVATE chess_core)

option(BUILD_TESTS "Build tests" ON)
if(BUILD_TESTS)
    enable_testing()
    include(FetchContent)
    FetchContent_Declare(
        googletest
        URL https://github.com/google/googletest/archive/refs/tags/v1.14.0.zip
    )
    set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
    FetchContent_MakeAvailable(googletest)

    add_executable(chess_tests
        tests/test_types.cpp
        tests/test_bitboard.cpp
        tests/test_attacks.cpp
        tests/test_position.cpp
        tests/test_movegen.cpp
        tests/test_perft.cpp
        tests/test_mcts.cpp
        tests/test_neural.cpp
    )
    target_link_libraries(chess_tests PRIVATE chess_core GTest::gtest_main)
    include(GoogleTest)
    gtest_discover_tests(chess_tests)
endif()
