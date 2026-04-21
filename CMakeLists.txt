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
    src/uci/stdout_responder.cpp
    src/uci/safety_filter.cpp
)
target_include_directories(chess_core PUBLIC src)

# Syzygy endgame tablebase support (Fathom, vendored under external/fathom).
# Builds unconditionally — the wrapper no-ops if TableBase::init() is never
# called, so there's no runtime cost when tablebases aren't present.
add_library(fathom STATIC external/fathom/src/tbprobe.c)
target_include_directories(fathom PUBLIC external/fathom/src)
set_target_properties(fathom PROPERTIES
    C_STANDARD 11
    C_STANDARD_REQUIRED YES
    C_EXTENSIONS OFF
)
if(MSVC)
    # Fathom's tbchess.c uses POSIX-isms; silence MSVC's CRT-secure warnings.
    target_compile_definitions(fathom PRIVATE _CRT_SECURE_NO_WARNINGS)
    # Fathom triggers /W4 noise on MSVC; quiet it down without affecting our code.
    target_compile_options(fathom PRIVATE /W0)
    # VS generator doesn't always honor C_STANDARD for C; force C11 explicitly
    # so <stdatomic.h> in tbprobe.c compiles. /experimental:c11atomics is still
    # required on MSVC 17.13 to actually enable the atomic_* types.
    target_compile_options(fathom PRIVATE /std:c11 /experimental:c11atomics)
endif()

target_sources(chess_core PRIVATE src/syzygy/syzygy.cpp)
target_link_libraries(chess_core PUBLIC fathom)

add_executable(chess_engine src/main.cpp)
target_link_libraries(chess_engine PRIVATE chess_core)

option(BUILD_TESTS "Build tests" ON)
if(BUILD_TESTS)
    enable_testing()
    include(FetchContent)
    FetchContent_Declare(
        googletest
        URL https://github.com/google/googletest/archive/refs/tags/v1.14.0.zip
        DOWNLOAD_EXTRACT_TIMESTAMP TRUE
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
        tests/test_syzygy.cpp
        tests/test_game_manager.cpp
    )
    target_link_libraries(chess_tests PRIVATE chess_core GTest::gtest_main)
    include(GoogleTest)
    # DISCOVERY_MODE PRE_TEST: enumerate test cases at `ctest` time, not build time.
    # At build time the exe can't always find its LibTorch/TRT DLLs (they get copied by
    # POST_BUILD steps that race with Ninja's discovery step). Deferring discovery sidesteps
    # the race and matches the VS generator's effective behavior.
    gtest_discover_tests(chess_tests
        WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
        DISCOVERY_MODE PRE_TEST
    )
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

    # Copy DLLs on Windows next to every exe/module that links libtorch.
    if(MSVC)
        file(GLOB TORCH_DLLS "${TORCH_INSTALL_PREFIX}/lib/*.dll")
        add_custom_command(TARGET chess_engine POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
            ${TORCH_DLLS} $<TARGET_FILE_DIR:chess_engine>)
        if(BUILD_TESTS)
            add_custom_command(TARGET chess_tests POST_BUILD
                COMMAND ${CMAKE_COMMAND} -E copy_if_different
                ${TORCH_DLLS} $<TARGET_FILE_DIR:chess_tests>)
        endif()
    endif()
endif()

# TensorRT backend (optional — requires TensorRT SDK + ENABLE_NEURAL)
option(ENABLE_TENSORRT "Build TensorRT inference backend" OFF)

if(ENABLE_TENSORRT)
    if(NOT ENABLE_NEURAL)
        message(FATAL_ERROR "ENABLE_TENSORRT requires ENABLE_NEURAL=ON (TRT evaluator reuses neural/BatchRequest/BatchResult types)")
    endif()

    if(DEFINED ENV{TENSORRT_PATH})
        set(TENSORRT_ROOT "$ENV{TENSORRT_PATH}")
    elseif(NOT TENSORRT_ROOT)
        message(FATAL_ERROR "ENABLE_TENSORRT=ON but TENSORRT_PATH env var / TENSORRT_ROOT CMake var is not set. Point it at your TensorRT SDK root (e.g. C:/TensorRT-10.16.1.11)")
    endif()

    find_path(TENSORRT_INCLUDE_DIR NvInfer.h HINTS "${TENSORRT_ROOT}/include" REQUIRED)
    find_library(TENSORRT_LIB_NVINFER     NAMES nvinfer_10 nvinfer  HINTS "${TENSORRT_ROOT}/lib" REQUIRED)
    find_library(TENSORRT_LIB_ONNXPARSER  NAMES nvonnxparser_10 nvonnxparser HINTS "${TENSORRT_ROOT}/lib" REQUIRED)

    target_sources(chess_neural PRIVATE src/neural/trt_evaluator.cpp)
    target_include_directories(chess_neural PUBLIC "${TENSORRT_INCLUDE_DIR}")
    target_link_libraries(chess_neural PUBLIC
        "${TENSORRT_LIB_NVINFER}"
        "${TENSORRT_LIB_ONNXPARSER}"
    )
    target_compile_definitions(chess_neural PUBLIC HAS_TENSORRT)
    target_compile_definitions(chess_engine PRIVATE HAS_TENSORRT)

    if(BUILD_TESTS)
        target_compile_definitions(chess_tests PRIVATE HAS_TENSORRT)
    endif()

    # Copy TRT DLLs on Windows alongside every exe/module that links TRT.
    if(MSVC)
        file(GLOB TRT_DLLS "${TENSORRT_ROOT}/bin/*.dll")
        add_custom_command(TARGET chess_engine POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
            ${TRT_DLLS} $<TARGET_FILE_DIR:chess_engine>)
        if(BUILD_TESTS)
            add_custom_command(TARGET chess_tests POST_BUILD
                COMMAND ${CMAKE_COMMAND} -E copy_if_different
                ${TRT_DLLS} $<TARGET_FILE_DIR:chess_tests>)
        endif()
    endif()

    message(STATUS "TensorRT: enabled (${TENSORRT_ROOT})")
endif()

# Python bindings (requires pybind11 and LibTorch)
option(BUILD_PYTHON "Build Python bindings" OFF)
if(BUILD_PYTHON AND ENABLE_NEURAL)
    find_package(Python COMPONENTS Interpreter Development.Module REQUIRED)
    find_package(pybind11 CONFIG REQUIRED)

    # The extension lives inside the `chess_mcts` Python package as `_core`.
    # User code does `import chess_mcts`; the package's __init__.py handles
    # DLL registration before re-exporting from ._core.
    pybind11_add_module(_core src/python/bindings.cpp)
    target_link_libraries(_core PRIVATE chess_core chess_neural)
    target_compile_definitions(_core PRIVATE HAS_LIBTORCH)

    if(ENABLE_TENSORRT)
        target_compile_definitions(_core PRIVATE HAS_TENSORRT)
    endif()

    # Bake LibTorch / TensorRT paths into _paths.py so chess_mcts/__init__.py
    # can register them with os.add_dll_directory on Windows. Paths are
    # captured at configure time from the same find_package() state the
    # extension itself was built against, so there's no environment drift.
    set(CHESS_MCTS_LIBTORCH_LIB "${TORCH_INSTALL_PREFIX}/lib")
    if(ENABLE_TENSORRT)
        set(CHESS_MCTS_TENSORRT_BIN "${TENSORRT_ROOT}/bin")
    else()
        set(CHESS_MCTS_TENSORRT_BIN "")
    endif()
    configure_file(
        "${CMAKE_SOURCE_DIR}/src/python/chess_mcts/_paths.py.in"
        "${CMAKE_CURRENT_BINARY_DIR}/chess_mcts_generated/_paths.py"
        @ONLY
    )

    # Install rules for scikit-build-core. The `python` component bundles
    # _core.pyd + _paths.py into the chess_mcts package directory inside
    # the wheel (or the editable-install redirect tree).
    install(TARGETS _core
            LIBRARY DESTINATION chess_mcts COMPONENT python
            RUNTIME DESTINATION chess_mcts COMPONENT python)
    install(FILES "${CMAKE_CURRENT_BINARY_DIR}/chess_mcts_generated/_paths.py"
            DESTINATION chess_mcts COMPONENT python)

    # build.ps1 / non-scikit-build path: keep copying DLLs next to the
    # extension for legacy direct-import workflows. Under scikit-build-core
    # the package's __init__.py handles DLL resolution via os.add_dll_directory,
    # so this copy is unnecessary (but harmless).
    if(MSVC AND NOT SKBUILD)
        file(GLOB TORCH_DLLS "${TORCH_INSTALL_PREFIX}/lib/*.dll")
        add_custom_command(TARGET _core POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
            ${TORCH_DLLS} $<TARGET_FILE_DIR:_core>)
        if(ENABLE_TENSORRT)
            file(GLOB TRT_DLLS "${TENSORRT_ROOT}/bin/*.dll")
            add_custom_command(TARGET _core POST_BUILD
                COMMAND ${CMAKE_COMMAND} -E copy_if_different
                ${TRT_DLLS} $<TARGET_FILE_DIR:_core>)
        endif()
    endif()
endif()
