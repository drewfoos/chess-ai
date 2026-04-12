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
    src/neural/encoder.cpp
    src/neural/position_history.cpp
    src/uci/uci.cpp
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
        tests/test_uci.cpp
    )
    target_link_libraries(chess_tests PRIVATE chess_core GTest::gtest_main)
    include(GoogleTest)
    gtest_discover_tests(chess_tests WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})
endif()

# Neural network support (optional — requires LibTorch)
option(ENABLE_NEURAL "Build neural network evaluator (requires LibTorch)" OFF)

if(ENABLE_NEURAL)
    find_package(Torch REQUIRED)

    add_library(chess_neural
        src/neural/neural_evaluator.cpp
        src/mcts/game_manager.cpp
    )
    target_include_directories(chess_neural PUBLIC src)
    target_link_libraries(chess_neural PUBLIC chess_core ${TORCH_LIBRARIES})
    target_compile_definitions(chess_neural PUBLIC HAS_LIBTORCH)

    target_link_libraries(chess_engine PRIVATE chess_neural)
    target_compile_definitions(chess_engine PRIVATE HAS_LIBTORCH)

    if(BUILD_TESTS)
        target_link_libraries(chess_tests PRIVATE chess_neural)
        target_compile_definitions(chess_tests PRIVATE HAS_LIBTORCH)
    endif()

    # Copy DLLs on Windows
    if(MSVC)
        file(GLOB TORCH_DLLS "${TORCH_INSTALL_PREFIX}/lib/*.dll")
        add_custom_command(TARGET chess_engine POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
            ${TORCH_DLLS} $<TARGET_FILE_DIR:chess_engine>)
    endif()
endif()

# Python bindings (requires pybind11 and LibTorch)
option(BUILD_PYTHON "Build Python bindings" OFF)
if(BUILD_PYTHON AND ENABLE_NEURAL)
    find_package(Python COMPONENTS Interpreter Development REQUIRED)
    find_package(pybind11 REQUIRED)
    pybind11_add_module(chess_mcts src/python/bindings.cpp)
    target_link_libraries(chess_mcts PRIVATE chess_core chess_neural)
    target_compile_definitions(chess_mcts PRIVATE HAS_LIBTORCH)

    # Copy DLLs on Windows
    if(MSVC)
        file(GLOB TORCH_DLLS "${TORCH_INSTALL_PREFIX}/lib/*.dll")
        add_custom_command(TARGET chess_mcts POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
            ${TORCH_DLLS} $<TARGET_FILE_DIR:chess_mcts>)
    endif()
endif()
