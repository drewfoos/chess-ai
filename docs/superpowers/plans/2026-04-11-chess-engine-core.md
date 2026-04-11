# Chess Engine Core — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a correct, tested chess engine core — bitboard board representation, full legal move generation, and make/unmake — validated by perft to known results.

**Architecture:** Bitboard-based board representation with piece-type + color arrays. Pseudo-legal move generation filtered for legality via king-in-check detection after make. Ray-based sliding piece attacks (no magic bitboards yet — optimization comes later). Unmake uses an undo-info stack.

**Tech Stack:** C++17, CMake 3.20+, Google Test (via FetchContent), MSVC (Visual Studio 2022 Community)

**Toolchain note:** This system has VS2022 Community but no CMake/g++ in PATH. All build commands use CMake via VS Developer Command Prompt or the bundled CMake. The plan includes setup instructions.

---

## Overall Plan Roadmap

This is **Plan 1 of 6** for the chess AI project:

| Plan | Subsystem | Depends On |
|------|-----------|------------|
| **1 (this)** | Chess Engine Core (types, bitboard, position, movegen, perft) | — |
| 2 | MCTS Search Engine | Plan 1 |
| 3 | Neural Network Architecture & Training Pipeline (Python/PyTorch) | — |
| 4 | Self-Play & Data Pipeline | Plans 1, 2 |
| 5 | C++ Neural Net Inference + Integration (closing the loop) | Plans 1–4 |
| 6 | Visualization Dashboard | Plans 1–5 |

Plans 1 and 3 can be developed in parallel.

---

## File Structure

```
chess-ai/
├── CMakeLists.txt                  # Root build config
├── src/
│   ├── core/
│   │   ├── types.h                 # Enums: Color, PieceType, Square, Move, CastlingRight
│   │   ├── bitboard.h              # Bitboard type, constants, inline utilities
│   │   ├── bitboard.cpp            # Bitboard print/debug functions
│   │   ├── attacks.h               # Attack table declarations
│   │   ├── attacks.cpp             # Attack table init + lookup functions
│   │   ├── position.h              # Position class declaration
│   │   ├── position.cpp            # Position implementation (FEN, make/unmake)
│   │   ├── movegen.h               # Move generation interface
│   │   └── movegen.cpp             # Move generation implementation
│   └── main.cpp                    # Placeholder entry point (perft CLI)
├── tests/
│   ├── test_types.cpp              # Move encoding/decoding tests
│   ├── test_bitboard.cpp           # Bitboard utility tests
│   ├── test_attacks.cpp            # Attack table correctness tests
│   ├── test_position.cpp           # FEN parsing, make/unmake tests
│   ├── test_movegen.cpp            # Move generation for specific positions
│   └── test_perft.cpp              # Perft validation against known results
└── docs/
    └── superpowers/plans/          # This plan lives here
```

---

## Task 1: Project Scaffolding

**Files:**
- Create: `CMakeLists.txt`
- Create: `src/main.cpp`
- Create: `src/core/types.h`
- Create: `tests/test_types.cpp`

### Step 1: Install CMake

- [ ] **Step 1a: Verify CMake is accessible**

VS2022 bundles CMake. Add it to PATH or use the full path. Run from Git Bash:

```bash
export PATH="/c/Program Files/Microsoft Visual Studio/2022/Community/Common7/IDE/CommonExtensions/Microsoft/CMake/CMake/bin:$PATH"
cmake --version
```

If this doesn't work, install CMake standalone: `winget install Kitware.CMake`

- [ ] **Step 1b: Verify MSVC compiler works**

```bash
# From Git Bash, test that CMake can find the MSVC compiler
mkdir -p /tmp/cmake-test && cd /tmp/cmake-test
cat > CMakeLists.txt << 'EOF'
cmake_minimum_required(VERSION 3.20)
project(test LANGUAGES CXX)
add_executable(test main.cpp)
EOF
cat > main.cpp << 'EOF'
#include <iostream>
int main() { std::cout << "works\n"; }
EOF
cmake -B build -G "Visual Studio 17 2022"
cmake --build build --config Release
./build/Release/test.exe
cd - && rm -rf /tmp/cmake-test
```

Expected: prints `works`

### Step 2: Create root CMakeLists.txt

- [ ] **Step 2a: Write CMakeLists.txt**

```cmake
# CMakeLists.txt
cmake_minimum_required(VERSION 3.20)
project(chess-ai VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Core chess library
add_library(chess_core
    src/core/bitboard.cpp
    src/core/attacks.cpp
    src/core/position.cpp
    src/core/movegen.cpp
)
target_include_directories(chess_core PUBLIC src)

# Main executable
add_executable(chess_engine src/main.cpp)
target_link_libraries(chess_engine PRIVATE chess_core)

# Tests
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
    )
    target_link_libraries(chess_tests PRIVATE chess_core GTest::gtest_main)
    include(GoogleTest)
    gtest_discover_tests(chess_tests)
endif()
```

### Step 3: Create types.h

- [ ] **Step 3a: Write the core type definitions**

```cpp
// src/core/types.h
#pragma once
#include <cstdint>
#include <string>

using Bitboard = uint64_t;

enum Color : uint8_t { WHITE, BLACK, NUM_COLORS };

constexpr Color operator~(Color c) { return Color(c ^ 1); }

enum PieceType : uint8_t {
    PAWN, KNIGHT, BISHOP, ROOK, QUEEN, KING,
    NUM_PIECE_TYPES,
    NO_PIECE_TYPE = 7
};

enum Square : uint8_t {
    A1, B1, C1, D1, E1, F1, G1, H1,
    A2, B2, C2, D2, E2, F2, G2, H2,
    A3, B3, C3, D3, E3, F3, G3, H3,
    A4, B4, C4, D4, E4, F4, G4, H4,
    A5, B5, C5, D5, E5, F5, G5, H5,
    A6, B6, C6, D6, E6, F6, G6, H6,
    A7, B7, C7, D7, E7, F7, G7, H7,
    A8, B8, C8, D8, E8, F8, G8, H8,
    NUM_SQUARES,
    NO_SQUARE = 64
};

constexpr int rank_of(Square s) { return s >> 3; }
constexpr int file_of(Square s) { return s & 7; }
constexpr Square make_square(int file, int rank) { return Square(rank * 8 + file); }

constexpr Square& operator++(Square& s) { return s = Square(int(s) + 1); }
constexpr Square operator+(Square s, int d) { return Square(int(s) + d); }
constexpr Square operator-(Square s, int d) { return Square(int(s) - d); }

enum MoveFlag : uint16_t {
    FLAG_QUIET          = 0,
    FLAG_DOUBLE_PUSH    = 1,
    FLAG_KING_CASTLE    = 2,
    FLAG_QUEEN_CASTLE   = 3,
    FLAG_CAPTURE        = 4,
    FLAG_EP_CAPTURE     = 5,
    // 6, 7 unused
    FLAG_PROMO_KNIGHT   = 8,
    FLAG_PROMO_BISHOP   = 9,
    FLAG_PROMO_ROOK     = 10,
    FLAG_PROMO_QUEEN    = 11,
    FLAG_PROMO_CAP_N    = 12,
    FLAG_PROMO_CAP_B    = 13,
    FLAG_PROMO_CAP_R    = 14,
    FLAG_PROMO_CAP_Q    = 15,
};

struct Move {
    uint16_t data;

    Move() : data(0) {}
    Move(Square from, Square to, MoveFlag flag = FLAG_QUIET)
        : data(uint16_t(from) | (uint16_t(to) << 6) | (uint16_t(flag) << 12)) {}

    Square from()     const { return Square(data & 0x3F); }
    Square to()       const { return Square((data >> 6) & 0x3F); }
    MoveFlag flag()   const { return MoveFlag(data >> 12); }

    bool is_capture()   const { return (flag() & 4) && !(flag() & 8) || (flag() >= FLAG_PROMO_CAP_N); }
    bool is_promotion() const { return flag() >= FLAG_PROMO_KNIGHT; }
    bool is_castle()    const { return flag() == FLAG_KING_CASTLE || flag() == FLAG_QUEEN_CASTLE; }
    bool is_ep()        const { return flag() == FLAG_EP_CAPTURE; }

    PieceType promo_piece() const { return PieceType((flag() & 3) + KNIGHT); }

    bool operator==(Move o) const { return data == o.data; }
    bool operator!=(Move o) const { return data != o.data; }

    // Null move sentinel
    static Move none() { return Move(); }
    bool is_none() const { return data == 0; }

    std::string to_uci() const;
};

enum CastlingRight : uint8_t {
    NO_CASTLING  = 0,
    WHITE_OO     = 1,
    WHITE_OOO    = 2,
    BLACK_OO     = 4,
    BLACK_OOO    = 8,
    WHITE_CASTLE = WHITE_OO | WHITE_OOO,
    BLACK_CASTLE = BLACK_OO | BLACK_OOO,
    ALL_CASTLING = 15
};

constexpr CastlingRight operator|(CastlingRight a, CastlingRight b) {
    return CastlingRight(uint8_t(a) | uint8_t(b));
}
constexpr CastlingRight operator&(CastlingRight a, CastlingRight b) {
    return CastlingRight(uint8_t(a) & uint8_t(b));
}
constexpr CastlingRight operator~(CastlingRight r) {
    return CastlingRight(~uint8_t(r) & 0xF);
}

// Direction offsets for ray generation
enum Direction : int {
    NORTH =  8, SOUTH = -8, EAST =  1, WEST = -1,
    NORTH_EAST = 9, NORTH_WEST = 7, SOUTH_EAST = -7, SOUTH_WEST = -9
};

// Maximum moves in any chess position (theoretical max ~218)
constexpr int MAX_MOVES = 256;
```

### Step 4: Write and run type tests

- [ ] **Step 4a: Write test_types.cpp**

```cpp
// tests/test_types.cpp
#include <gtest/gtest.h>
#include "core/types.h"

TEST(Types, SquareCoordinates) {
    EXPECT_EQ(rank_of(A1), 0);
    EXPECT_EQ(file_of(A1), 0);
    EXPECT_EQ(rank_of(H8), 7);
    EXPECT_EQ(file_of(H8), 7);
    EXPECT_EQ(make_square(4, 3), E4);
    EXPECT_EQ(make_square(0, 7), A8);
}

TEST(Types, ColorFlip) {
    EXPECT_EQ(~WHITE, BLACK);
    EXPECT_EQ(~BLACK, WHITE);
}

TEST(Types, MoveEncodeDecode) {
    Move m(E2, E4, FLAG_DOUBLE_PUSH);
    EXPECT_EQ(m.from(), E2);
    EXPECT_EQ(m.to(), E4);
    EXPECT_EQ(m.flag(), FLAG_DOUBLE_PUSH);
    EXPECT_FALSE(m.is_capture());
    EXPECT_FALSE(m.is_promotion());
}

TEST(Types, MoveCapture) {
    Move m(D4, E5, FLAG_CAPTURE);
    EXPECT_TRUE(m.is_capture());
    EXPECT_FALSE(m.is_promotion());
}

TEST(Types, MovePromotion) {
    Move m(A7, A8, FLAG_PROMO_QUEEN);
    EXPECT_TRUE(m.is_promotion());
    EXPECT_FALSE(m.is_capture());
    EXPECT_EQ(m.promo_piece(), QUEEN);
}

TEST(Types, MovePromotionCapture) {
    Move m(B7, A8, FLAG_PROMO_CAP_KNIGHT);
    EXPECT_TRUE(m.is_promotion());
    EXPECT_TRUE(m.is_capture());
    EXPECT_EQ(m.promo_piece(), KNIGHT);
}

TEST(Types, MoveCastle) {
    Move m(E1, G1, FLAG_KING_CASTLE);
    EXPECT_TRUE(m.is_castle());
    EXPECT_FALSE(m.is_capture());
}

TEST(Types, MoveNone) {
    EXPECT_TRUE(Move::none().is_none());
    Move m(A2, A3);
    EXPECT_FALSE(m.is_none());
}

TEST(Types, CastlingBitOps) {
    CastlingRight cr = WHITE_OO | BLACK_OOO;
    EXPECT_EQ(cr & WHITE_OO, WHITE_OO);
    EXPECT_EQ(cr & WHITE_OOO, NO_CASTLING);
    EXPECT_EQ(cr & BLACK_OOO, BLACK_OOO);
}
```

- [ ] **Step 4b: Create stub source files so CMake doesn't fail**

```cpp
// src/core/bitboard.cpp
#include "core/bitboard.h"

// src/core/attacks.cpp
#include "core/attacks.h"

// src/core/position.cpp
#include "core/position.h"

// src/core/movegen.cpp
#include "core/movegen.h"

// src/main.cpp
int main() { return 0; }
```

Create minimal stub headers too (will be filled in later tasks):

```cpp
// src/core/bitboard.h
#pragma once
#include "core/types.h"

// src/core/attacks.h
#pragma once
#include "core/bitboard.h"

// src/core/position.h
#pragma once
#include "core/types.h"
#include "core/bitboard.h"

// src/core/movegen.h
#pragma once
#include "core/position.h"
```

And stub test files:

```cpp
// tests/test_bitboard.cpp
#include <gtest/gtest.h>
TEST(Bitboard, Placeholder) { EXPECT_TRUE(true); }

// tests/test_attacks.cpp
#include <gtest/gtest.h>
TEST(Attacks, Placeholder) { EXPECT_TRUE(true); }

// tests/test_position.cpp
#include <gtest/gtest.h>
TEST(Position, Placeholder) { EXPECT_TRUE(true); }

// tests/test_movegen.cpp
#include <gtest/gtest.h>
TEST(MoveGen, Placeholder) { EXPECT_TRUE(true); }

// tests/test_perft.cpp
#include <gtest/gtest.h>
TEST(Perft, Placeholder) { EXPECT_TRUE(true); }
```

- [ ] **Step 4c: Build and run tests**

```bash
cd E:/dev/chess-ai
cmake -B build -G "Visual Studio 17 2022"
cmake --build build --config Release
ctest --test-dir build --build-config Release --output-on-failure
```

Expected: All tests pass (type tests + placeholder tests).

- [ ] **Step 4d: Commit**

```bash
git init
git add CMakeLists.txt src/ tests/ docs/
git commit -m "feat: project scaffolding with types, CMake, and Google Test"
```

---

## Task 2: Bitboard Utilities

**Files:**
- Modify: `src/core/bitboard.h`
- Modify: `src/core/bitboard.cpp`
- Replace: `tests/test_bitboard.cpp`

### Step 1: Write bitboard tests

- [ ] **Step 1a: Write test_bitboard.cpp**

```cpp
// tests/test_bitboard.cpp
#include <gtest/gtest.h>
#include "core/bitboard.h"

TEST(Bitboard, SquareBB) {
    EXPECT_EQ(square_bb(A1), 1ULL);
    EXPECT_EQ(square_bb(H1), 1ULL << 7);
    EXPECT_EQ(square_bb(A8), 1ULL << 56);
    EXPECT_EQ(square_bb(H8), 1ULL << 63);
}

TEST(Bitboard, RankFileBB) {
    // Rank 1 = bottom row
    EXPECT_EQ(RANK_BB[0], 0xFFULL);
    // Rank 8 = top row
    EXPECT_EQ(RANK_BB[7], 0xFFULL << 56);
    // File A = leftmost column
    EXPECT_EQ(FILE_BB[0], 0x0101010101010101ULL);
    // File H = rightmost column
    EXPECT_EQ(FILE_BB[7], 0x8080808080808080ULL);
}

TEST(Bitboard, Popcount) {
    EXPECT_EQ(popcount(0ULL), 0);
    EXPECT_EQ(popcount(0xFFULL), 8);
    EXPECT_EQ(popcount(0xFFFFFFFFFFFFFFFFULL), 64);
    EXPECT_EQ(popcount(square_bb(E4)), 1);
}

TEST(Bitboard, LSB) {
    EXPECT_EQ(lsb(1ULL), A1);
    EXPECT_EQ(lsb(1ULL << 63), H8);
    EXPECT_EQ(lsb(0x0100000000ULL), A5);
}

TEST(Bitboard, PopLSB) {
    Bitboard bb = square_bb(A1) | square_bb(C3) | square_bb(H8);
    Square s1 = pop_lsb(bb);
    EXPECT_EQ(s1, A1);
    Square s2 = pop_lsb(bb);
    EXPECT_EQ(s2, C3);
    Square s3 = pop_lsb(bb);
    EXPECT_EQ(s3, H8);
    EXPECT_EQ(bb, 0ULL);
}

TEST(Bitboard, ShiftNorth) {
    Bitboard rank1 = RANK_BB[0];
    EXPECT_EQ(shift_bb<NORTH>(rank1), RANK_BB[1]);
}

TEST(Bitboard, ShiftSouth) {
    Bitboard rank2 = RANK_BB[1];
    EXPECT_EQ(shift_bb<SOUTH>(rank2), RANK_BB[0]);
}

TEST(Bitboard, ShiftEastWraparound) {
    // East shift should not wrap from H-file to A-file
    Bitboard h_file = FILE_BB[7];
    EXPECT_EQ(shift_bb<EAST>(h_file), 0ULL);
}

TEST(Bitboard, ShiftWestWraparound) {
    Bitboard a_file = FILE_BB[0];
    EXPECT_EQ(shift_bb<WEST>(a_file), 0ULL);
}

TEST(Bitboard, MultipleBits) {
    Bitboard bb = square_bb(A1) | square_bb(B2) | square_bb(C3);
    EXPECT_EQ(popcount(bb), 3);
    EXPECT_TRUE(bb & square_bb(B2));
    EXPECT_FALSE(bb & square_bb(D4));
}
```

- [ ] **Step 1b: Run tests — verify they fail**

```bash
cmake --build build --config Release 2>&1
```

Expected: Compile error — `square_bb`, `RANK_BB`, etc. not defined.

### Step 2: Implement bitboard.h

- [ ] **Step 2a: Write bitboard.h**

```cpp
// src/core/bitboard.h
#pragma once
#include "core/types.h"
#include <string>

#ifdef _MSC_VER
#include <intrin.h>
#endif

// --- Constants ---

constexpr Bitboard FILE_A_BB = 0x0101010101010101ULL;
constexpr Bitboard FILE_H_BB = 0x8080808080808080ULL;

constexpr Bitboard FILE_BB[8] = {
    FILE_A_BB, FILE_A_BB << 1, FILE_A_BB << 2, FILE_A_BB << 3,
    FILE_A_BB << 4, FILE_A_BB << 5, FILE_A_BB << 6, FILE_A_BB << 7,
};

constexpr Bitboard RANK_1_BB = 0xFFULL;

constexpr Bitboard RANK_BB[8] = {
    RANK_1_BB, RANK_1_BB << 8, RANK_1_BB << 16, RANK_1_BB << 24,
    RANK_1_BB << 32, RANK_1_BB << 40, RANK_1_BB << 48, RANK_1_BB << 56,
};

// --- Square bitboard ---

constexpr Bitboard square_bb(Square s) { return 1ULL << s; }

// --- Bit manipulation ---

inline int popcount(Bitboard bb) {
#ifdef _MSC_VER
    return int(__popcnt64(bb));
#else
    return __builtin_popcountll(bb);
#endif
}

inline Square lsb(Bitboard bb) {
#ifdef _MSC_VER
    unsigned long idx;
    _BitScanForward64(&idx, bb);
    return Square(idx);
#else
    return Square(__builtin_ctzll(bb));
#endif
}

inline Square pop_lsb(Bitboard& bb) {
    Square s = lsb(bb);
    bb &= bb - 1;
    return s;
}

// --- Shifts (with wrapping prevention) ---

template<Direction D>
constexpr Bitboard shift_bb(Bitboard bb) {
    if constexpr (D == NORTH)      return bb << 8;
    if constexpr (D == SOUTH)      return bb >> 8;
    if constexpr (D == EAST)       return (bb & ~FILE_H_BB) << 1;
    if constexpr (D == WEST)       return (bb & ~FILE_A_BB) >> 1;
    if constexpr (D == NORTH_EAST) return (bb & ~FILE_H_BB) << 9;
    if constexpr (D == NORTH_WEST) return (bb & ~FILE_A_BB) << 7;
    if constexpr (D == SOUTH_EAST) return (bb & ~FILE_H_BB) >> 7;
    if constexpr (D == SOUTH_WEST) return (bb & ~FILE_A_BB) >> 9;
    return 0;
}

// --- Debug ---

std::string bitboard_to_string(Bitboard bb);
```

- [ ] **Step 2b: Write bitboard.cpp**

```cpp
// src/core/bitboard.cpp
#include "core/bitboard.h"
#include <sstream>

std::string bitboard_to_string(Bitboard bb) {
    std::ostringstream ss;
    for (int rank = 7; rank >= 0; --rank) {
        for (int file = 0; file < 8; ++file) {
            Square s = make_square(file, rank);
            ss << ((bb & square_bb(s)) ? '1' : '.') << ' ';
        }
        ss << '\n';
    }
    return ss.str();
}
```

- [ ] **Step 2c: Build and run tests**

```bash
cmake --build build --config Release
ctest --test-dir build --build-config Release --output-on-failure -R Bitboard
```

Expected: All Bitboard tests pass.

- [ ] **Step 2d: Commit**

```bash
git add src/core/bitboard.h src/core/bitboard.cpp tests/test_bitboard.cpp
git commit -m "feat: bitboard utilities — square_bb, popcount, lsb, shifts"
```

---

## Task 3: Attack Tables

**Files:**
- Modify: `src/core/attacks.h`
- Modify: `src/core/attacks.cpp`
- Replace: `tests/test_attacks.cpp`

### Step 1: Write attack table tests

- [ ] **Step 1a: Write test_attacks.cpp**

```cpp
// tests/test_attacks.cpp
#include <gtest/gtest.h>
#include "core/attacks.h"

class AttacksTest : public ::testing::Test {
protected:
    void SetUp() override {
        attacks::init();
    }
};

// Knight attacks
TEST_F(AttacksTest, KnightCenter) {
    // Knight on E4 attacks 8 squares
    Bitboard atk = attacks::knight(E4);
    EXPECT_EQ(popcount(atk), 8);
    EXPECT_TRUE(atk & square_bb(D6));
    EXPECT_TRUE(atk & square_bb(F6));
    EXPECT_TRUE(atk & square_bb(G5));
    EXPECT_TRUE(atk & square_bb(G3));
    EXPECT_TRUE(atk & square_bb(F2));
    EXPECT_TRUE(atk & square_bb(D2));
    EXPECT_TRUE(atk & square_bb(C3));
    EXPECT_TRUE(atk & square_bb(C5));
}

TEST_F(AttacksTest, KnightCorner) {
    // Knight on A1 attacks 2 squares
    Bitboard atk = attacks::knight(A1);
    EXPECT_EQ(popcount(atk), 2);
    EXPECT_TRUE(atk & square_bb(B3));
    EXPECT_TRUE(atk & square_bb(C2));
}

// King attacks
TEST_F(AttacksTest, KingCenter) {
    Bitboard atk = attacks::king(E4);
    EXPECT_EQ(popcount(atk), 8);
    EXPECT_TRUE(atk & square_bb(D5));
    EXPECT_TRUE(atk & square_bb(E5));
    EXPECT_TRUE(atk & square_bb(F5));
    EXPECT_TRUE(atk & square_bb(D4));
    EXPECT_TRUE(atk & square_bb(F4));
    EXPECT_TRUE(atk & square_bb(D3));
    EXPECT_TRUE(atk & square_bb(E3));
    EXPECT_TRUE(atk & square_bb(F3));
}

TEST_F(AttacksTest, KingCorner) {
    Bitboard atk = attacks::king(A1);
    EXPECT_EQ(popcount(atk), 3);
}

// Pawn attacks
TEST_F(AttacksTest, WhitePawnCenter) {
    Bitboard atk = attacks::pawn(WHITE, E4);
    EXPECT_EQ(popcount(atk), 2);
    EXPECT_TRUE(atk & square_bb(D5));
    EXPECT_TRUE(atk & square_bb(F5));
}

TEST_F(AttacksTest, WhitePawnAFile) {
    Bitboard atk = attacks::pawn(WHITE, A4);
    EXPECT_EQ(popcount(atk), 1);
    EXPECT_TRUE(atk & square_bb(B5));
}

TEST_F(AttacksTest, BlackPawnCenter) {
    Bitboard atk = attacks::pawn(BLACK, E5);
    EXPECT_EQ(popcount(atk), 2);
    EXPECT_TRUE(atk & square_bb(D4));
    EXPECT_TRUE(atk & square_bb(F4));
}

// Sliding piece attacks
TEST_F(AttacksTest, BishopOpenBoard) {
    // Bishop on E4 with no blockers: attacks along diagonals
    Bitboard atk = attacks::bishop(E4, 0ULL);
    EXPECT_TRUE(atk & square_bb(D5));
    EXPECT_TRUE(atk & square_bb(A8)); // long diagonal
    EXPECT_TRUE(atk & square_bb(H7)); // other diagonal
    EXPECT_TRUE(atk & square_bb(H1));
    EXPECT_TRUE(atk & square_bb(B1));
    EXPECT_FALSE(atk & square_bb(E5)); // not on rank/file
}

TEST_F(AttacksTest, BishopBlocker) {
    // Piece on F5 should stop the ray but be included (could be capture)
    Bitboard occ = square_bb(F5);
    Bitboard atk = attacks::bishop(E4, occ);
    EXPECT_TRUE(atk & square_bb(F5));   // blocker square included
    EXPECT_FALSE(atk & square_bb(G6));  // past blocker: not included
}

TEST_F(AttacksTest, RookOpenBoard) {
    Bitboard atk = attacks::rook(E4, 0ULL);
    // Full rank and file minus the square itself
    EXPECT_TRUE(atk & square_bb(E1));
    EXPECT_TRUE(atk & square_bb(E8));
    EXPECT_TRUE(atk & square_bb(A4));
    EXPECT_TRUE(atk & square_bb(H4));
    EXPECT_FALSE(atk & square_bb(D5)); // diagonal: no
}

TEST_F(AttacksTest, RookBlocker) {
    Bitboard occ = square_bb(E6);
    Bitboard atk = attacks::rook(E4, occ);
    EXPECT_TRUE(atk & square_bb(E5));
    EXPECT_TRUE(atk & square_bb(E6));   // blocker included
    EXPECT_FALSE(atk & square_bb(E7));  // past blocker
}

TEST_F(AttacksTest, QueenIsRookPlusBishop) {
    Bitboard occ = square_bb(G6) | square_bb(C2);
    Bitboard q = attacks::queen(E4, occ);
    Bitboard r = attacks::rook(E4, occ);
    Bitboard b = attacks::bishop(E4, occ);
    EXPECT_EQ(q, r | b);
}
```

- [ ] **Step 1b: Run tests — verify they fail**

```bash
cmake --build build --config Release 2>&1
```

Expected: Compile error — `attacks::` namespace not defined.

### Step 2: Implement attack tables

- [ ] **Step 2a: Write attacks.h**

```cpp
// src/core/attacks.h
#pragma once
#include "core/bitboard.h"

namespace attacks {

// Must be called once at startup before using any attack functions
void init();

// Precomputed (no occupancy needed)
Bitboard knight(Square s);
Bitboard king(Square s);
Bitboard pawn(Color c, Square s);

// Occupancy-dependent (ray-based iteration, no magic bitboards)
Bitboard bishop(Square s, Bitboard occupied);
Bitboard rook(Square s, Bitboard occupied);
Bitboard queen(Square s, Bitboard occupied);

} // namespace attacks
```

- [ ] **Step 2b: Write attacks.cpp**

```cpp
// src/core/attacks.cpp
#include "core/attacks.h"

namespace attacks {

static Bitboard knight_attacks[NUM_SQUARES];
static Bitboard king_attacks[NUM_SQUARES];
static Bitboard pawn_attacks[NUM_COLORS][NUM_SQUARES];

// Ray in a single direction from a square, stopping at board edge
static Bitboard ray_attacks(Square s, Direction d, Bitboard occupied) {
    Bitboard attacks = 0;
    Bitboard bb = square_bb(s);

    while (true) {
        // Shift one step in direction
        if (d == NORTH)      bb <<= 8;
        else if (d == SOUTH) bb >>= 8;
        else if (d == EAST)       { if (bb & FILE_H_BB) break; bb <<= 1; }
        else if (d == WEST)       { if (bb & FILE_A_BB) break; bb >>= 1; }
        else if (d == NORTH_EAST) { if (bb & FILE_H_BB) break; bb <<= 9; }
        else if (d == NORTH_WEST) { if (bb & FILE_A_BB) break; bb <<= 7; }
        else if (d == SOUTH_EAST) { if (bb & FILE_H_BB) break; bb >>= 7; }
        else if (d == SOUTH_WEST) { if (bb & FILE_A_BB) break; bb >>= 9; }

        if (bb == 0) break;

        attacks |= bb;

        if (bb & occupied) break; // Hit a piece — include it, then stop
    }

    return attacks;
}

static void init_knight_attacks() {
    const int offsets[8][2] = {
        {-2,-1},{-2,1},{-1,-2},{-1,2},{1,-2},{1,2},{2,-1},{2,1}
    };
    for (int sq = 0; sq < 64; ++sq) {
        Bitboard bb = 0;
        int r = rank_of(Square(sq));
        int f = file_of(Square(sq));
        for (auto& [dr, df] : offsets) {
            int nr = r + dr, nf = f + df;
            if (nr >= 0 && nr < 8 && nf >= 0 && nf < 8)
                bb |= square_bb(make_square(nf, nr));
        }
        knight_attacks[sq] = bb;
    }
}

static void init_king_attacks() {
    const int offsets[8][2] = {
        {-1,-1},{-1,0},{-1,1},{0,-1},{0,1},{1,-1},{1,0},{1,1}
    };
    for (int sq = 0; sq < 64; ++sq) {
        Bitboard bb = 0;
        int r = rank_of(Square(sq));
        int f = file_of(Square(sq));
        for (auto& [dr, df] : offsets) {
            int nr = r + dr, nf = f + df;
            if (nr >= 0 && nr < 8 && nf >= 0 && nf < 8)
                bb |= square_bb(make_square(nf, nr));
        }
        king_attacks[sq] = bb;
    }
}

static void init_pawn_attacks() {
    for (int sq = 0; sq < 64; ++sq) {
        int r = rank_of(Square(sq));
        int f = file_of(Square(sq));

        Bitboard white_atk = 0, black_atk = 0;

        // White pawns attack NE and NW
        if (r < 7) {
            if (f > 0) white_atk |= square_bb(make_square(f - 1, r + 1));
            if (f < 7) white_atk |= square_bb(make_square(f + 1, r + 1));
        }

        // Black pawns attack SE and SW
        if (r > 0) {
            if (f > 0) black_atk |= square_bb(make_square(f - 1, r - 1));
            if (f < 7) black_atk |= square_bb(make_square(f + 1, r - 1));
        }

        pawn_attacks[WHITE][sq] = white_atk;
        pawn_attacks[BLACK][sq] = black_atk;
    }
}

void init() {
    init_knight_attacks();
    init_king_attacks();
    init_pawn_attacks();
}

Bitboard knight(Square s) { return knight_attacks[s]; }
Bitboard king(Square s)   { return king_attacks[s]; }
Bitboard pawn(Color c, Square s) { return pawn_attacks[c][s]; }

Bitboard bishop(Square s, Bitboard occupied) {
    return ray_attacks(s, NORTH_EAST, occupied)
         | ray_attacks(s, NORTH_WEST, occupied)
         | ray_attacks(s, SOUTH_EAST, occupied)
         | ray_attacks(s, SOUTH_WEST, occupied);
}

Bitboard rook(Square s, Bitboard occupied) {
    return ray_attacks(s, NORTH, occupied)
         | ray_attacks(s, SOUTH, occupied)
         | ray_attacks(s, EAST,  occupied)
         | ray_attacks(s, WEST,  occupied);
}

Bitboard queen(Square s, Bitboard occupied) {
    return bishop(s, occupied) | rook(s, occupied);
}

} // namespace attacks
```

- [ ] **Step 2c: Build and run tests**

```bash
cmake --build build --config Release
ctest --test-dir build --build-config Release --output-on-failure -R Attacks
```

Expected: All Attacks tests pass.

- [ ] **Step 2d: Commit**

```bash
git add src/core/attacks.h src/core/attacks.cpp tests/test_attacks.cpp
git commit -m "feat: attack tables — knight, king, pawn, bishop, rook, queen"
```

---

## Task 4: Position Class & FEN Parsing

**Files:**
- Modify: `src/core/position.h`
- Modify: `src/core/position.cpp`
- Modify: `src/core/types.h` (add `Move::to_uci()`)
- Replace: `tests/test_position.cpp`

### Step 1: Write position tests

- [ ] **Step 1a: Write test_position.cpp**

```cpp
// tests/test_position.cpp
#include <gtest/gtest.h>
#include "core/position.h"
#include "core/attacks.h"

class PositionTest : public ::testing::Test {
protected:
    void SetUp() override {
        attacks::init();
    }
};

TEST_F(PositionTest, StartingPositionFEN) {
    Position pos;
    pos.set_fen("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1");

    EXPECT_EQ(pos.side_to_move(), WHITE);
    EXPECT_EQ(pos.castling_rights(), ALL_CASTLING);
    EXPECT_EQ(pos.ep_square(), NO_SQUARE);
    EXPECT_EQ(pos.halfmove_clock(), 0);
    EXPECT_EQ(pos.fullmove_number(), 1);

    // Check specific pieces
    EXPECT_EQ(pos.piece_on(E1), KING);
    EXPECT_EQ(pos.color_on(E1), WHITE);
    EXPECT_EQ(pos.piece_on(E8), KING);
    EXPECT_EQ(pos.color_on(E8), BLACK);
    EXPECT_EQ(pos.piece_on(A2), PAWN);
    EXPECT_EQ(pos.piece_on(D8), QUEEN);
    EXPECT_EQ(pos.piece_on(E4), NO_PIECE_TYPE);
}

TEST_F(PositionTest, FENRoundTrip) {
    const char* fens[] = {
        "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
        "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1",
        "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
        "8/8/8/8/8/8/8/4K2k w - - 0 1",
        "r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1",
    };

    Position pos;
    for (const char* fen : fens) {
        pos.set_fen(fen);
        EXPECT_EQ(pos.to_fen(), std::string(fen)) << "Failed roundtrip for: " << fen;
    }
}

TEST_F(PositionTest, PieceBitboards) {
    Position pos;
    pos.set_fen("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1");

    // White pawns on rank 2
    Bitboard wp = pos.pieces(WHITE, PAWN);
    EXPECT_EQ(wp, RANK_BB[1]);

    // Black pawns on rank 7
    Bitboard bp = pos.pieces(BLACK, PAWN);
    EXPECT_EQ(bp, RANK_BB[6]);

    // Total occupancy = 32 pieces
    EXPECT_EQ(popcount(pos.occupied()), 32);
    EXPECT_EQ(popcount(pos.occupied(WHITE)), 16);
    EXPECT_EQ(popcount(pos.occupied(BLACK)), 16);
}

TEST_F(PositionTest, IsSquareAttacked) {
    Position pos;
    pos.set_fen("rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1");

    // E4 pawn attacks D5 and F5
    EXPECT_TRUE(pos.is_attacked(D5, WHITE));
    EXPECT_TRUE(pos.is_attacked(F5, WHITE));

    // E5 is not attacked by white in starting-ish position
    EXPECT_FALSE(pos.is_attacked(E5, WHITE));
}

TEST_F(PositionTest, InCheck) {
    Position pos;

    // Not in check in starting position
    pos.set_fen("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1");
    EXPECT_FALSE(pos.in_check());

    // Simple check: black king on E8 attacked by white rook on E1
    pos.set_fen("4k3/8/8/8/8/8/8/4R2K b - - 0 1");
    EXPECT_TRUE(pos.in_check());
}
```

- [ ] **Step 1b: Run tests — verify they fail**

```bash
cmake --build build --config Release 2>&1
```

Expected: Compile error — `Position` class not defined.

### Step 2: Implement Position class

- [ ] **Step 2a: Write position.h**

```cpp
// src/core/position.h
#pragma once
#include "core/types.h"
#include "core/bitboard.h"
#include <string>
#include <vector>

struct UndoInfo {
    CastlingRight castling;
    Square ep;
    int halfmove;
    PieceType captured;
};

class Position {
public:
    Position() = default;

    // Setup
    void set_fen(const std::string& fen);
    std::string to_fen() const;

    // Accessors
    Color side_to_move()      const { return side_to_move_; }
    CastlingRight castling_rights() const { return castling_; }
    Square ep_square()        const { return ep_square_; }
    int halfmove_clock()      const { return halfmove_clock_; }
    int fullmove_number()     const { return fullmove_number_; }

    PieceType piece_on(Square s) const { return board_[s]; }
    Color color_on(Square s)     const { return color_[s]; }

    Bitboard pieces(Color c, PieceType pt) const { return bb_pieces_[c][pt]; }
    Bitboard pieces(Color c)               const { return bb_color_[c]; }
    Bitboard occupied()                    const { return bb_color_[WHITE] | bb_color_[BLACK]; }
    Bitboard occupied(Color c)             const { return bb_color_[c]; }

    Square king_square(Color c) const;

    // Attack detection
    bool is_attacked(Square s, Color by) const;
    bool in_check() const;
    Bitboard attackers_to(Square s, Bitboard occ) const;

    // Make/unmake
    void make_move(Move m, UndoInfo& undo);
    void unmake_move(Move m, const UndoInfo& undo);

private:
    // Piece placement
    Bitboard bb_pieces_[NUM_COLORS][NUM_PIECE_TYPES] = {};
    Bitboard bb_color_[NUM_COLORS] = {};
    PieceType board_[NUM_SQUARES] = {};
    Color color_[NUM_SQUARES] = {};

    // State
    Color side_to_move_ = WHITE;
    CastlingRight castling_ = NO_CASTLING;
    Square ep_square_ = NO_SQUARE;
    int halfmove_clock_ = 0;
    int fullmove_number_ = 1;

    // Helpers
    void put_piece(Color c, PieceType pt, Square s);
    void remove_piece(Square s);
    void move_piece(Square from, Square to);
};
```

- [ ] **Step 2b: Write position.cpp**

```cpp
// src/core/position.cpp
#include "core/position.h"
#include "core/attacks.h"
#include <sstream>

void Position::put_piece(Color c, PieceType pt, Square s) {
    Bitboard bb = square_bb(s);
    bb_pieces_[c][pt] |= bb;
    bb_color_[c] |= bb;
    board_[s] = pt;
    color_[s] = c;
}

void Position::remove_piece(Square s) {
    Bitboard bb = square_bb(s);
    Color c = color_[s];
    PieceType pt = board_[s];
    bb_pieces_[c][pt] ^= bb;
    bb_color_[c] ^= bb;
    board_[s] = NO_PIECE_TYPE;
}

void Position::move_piece(Square from, Square to) {
    Bitboard from_to = square_bb(from) | square_bb(to);
    Color c = color_[from];
    PieceType pt = board_[from];
    bb_pieces_[c][pt] ^= from_to;
    bb_color_[c] ^= from_to;
    board_[to] = pt;
    color_[to] = c;
    board_[from] = NO_PIECE_TYPE;
}

Square Position::king_square(Color c) const {
    return lsb(bb_pieces_[c][KING]);
}

void Position::set_fen(const std::string& fen) {
    // Clear board
    for (int i = 0; i < NUM_SQUARES; ++i) board_[i] = NO_PIECE_TYPE;
    for (int c = 0; c < NUM_COLORS; ++c) {
        bb_color_[c] = 0;
        for (int pt = 0; pt < NUM_PIECE_TYPES; ++pt)
            bb_pieces_[c][pt] = 0;
    }

    std::istringstream ss(fen);
    std::string pieces, side, castling, ep;
    int hm, fm;
    ss >> pieces >> side >> castling >> ep >> hm >> fm;

    // Parse piece placement
    int rank = 7, file = 0;
    for (char ch : pieces) {
        if (ch == '/') { --rank; file = 0; }
        else if (ch >= '1' && ch <= '8') { file += ch - '0'; }
        else {
            Color c = (ch >= 'A' && ch <= 'Z') ? WHITE : BLACK;
            char lower = ch | 0x20; // to lowercase
            PieceType pt;
            switch (lower) {
                case 'p': pt = PAWN;   break;
                case 'n': pt = KNIGHT; break;
                case 'b': pt = BISHOP; break;
                case 'r': pt = ROOK;   break;
                case 'q': pt = QUEEN;  break;
                case 'k': pt = KING;   break;
                default:  pt = NO_PIECE_TYPE; break;
            }
            put_piece(c, pt, make_square(file, rank));
            ++file;
        }
    }

    // Side to move
    side_to_move_ = (side == "w") ? WHITE : BLACK;

    // Castling rights
    castling_ = NO_CASTLING;
    if (castling != "-") {
        for (char ch : castling) {
            if (ch == 'K') castling_ = castling_ | WHITE_OO;
            if (ch == 'Q') castling_ = castling_ | WHITE_OOO;
            if (ch == 'k') castling_ = castling_ | BLACK_OO;
            if (ch == 'q') castling_ = castling_ | BLACK_OOO;
        }
    }

    // En passant
    if (ep == "-") {
        ep_square_ = NO_SQUARE;
    } else {
        int f = ep[0] - 'a';
        int r = ep[1] - '1';
        ep_square_ = make_square(f, r);
    }

    halfmove_clock_ = hm;
    fullmove_number_ = fm;
}

std::string Position::to_fen() const {
    std::ostringstream ss;
    const char piece_chars[] = "pnbrqk";

    for (int rank = 7; rank >= 0; --rank) {
        int empty = 0;
        for (int file = 0; file < 8; ++file) {
            Square s = make_square(file, rank);
            if (board_[s] == NO_PIECE_TYPE) {
                ++empty;
            } else {
                if (empty > 0) { ss << empty; empty = 0; }
                char ch = piece_chars[board_[s]];
                if (color_[s] == WHITE) ch &= ~0x20; // uppercase
                ss << ch;
            }
        }
        if (empty > 0) ss << empty;
        if (rank > 0) ss << '/';
    }

    ss << (side_to_move_ == WHITE ? " w " : " b ");

    if (castling_ == NO_CASTLING) {
        ss << '-';
    } else {
        if (castling_ & WHITE_OO)  ss << 'K';
        if (castling_ & WHITE_OOO) ss << 'Q';
        if (castling_ & BLACK_OO)  ss << 'k';
        if (castling_ & BLACK_OOO) ss << 'q';
    }

    ss << ' ';
    if (ep_square_ == NO_SQUARE) {
        ss << '-';
    } else {
        ss << char('a' + file_of(ep_square_)) << char('1' + rank_of(ep_square_));
    }

    ss << ' ' << halfmove_clock_ << ' ' << fullmove_number_;
    return ss.str();
}

Bitboard Position::attackers_to(Square s, Bitboard occ) const {
    return (attacks::pawn(BLACK, s) & bb_pieces_[WHITE][PAWN])
         | (attacks::pawn(WHITE, s) & bb_pieces_[BLACK][PAWN])
         | (attacks::knight(s)      & (bb_pieces_[WHITE][KNIGHT] | bb_pieces_[BLACK][KNIGHT]))
         | (attacks::bishop(s, occ) & (bb_pieces_[WHITE][BISHOP] | bb_pieces_[BLACK][BISHOP]
                                     | bb_pieces_[WHITE][QUEEN]  | bb_pieces_[BLACK][QUEEN]))
         | (attacks::rook(s, occ)   & (bb_pieces_[WHITE][ROOK]   | bb_pieces_[BLACK][ROOK]
                                     | bb_pieces_[WHITE][QUEEN]  | bb_pieces_[BLACK][QUEEN]))
         | (attacks::king(s)        & (bb_pieces_[WHITE][KING]   | bb_pieces_[BLACK][KING]));
}

bool Position::is_attacked(Square s, Color by) const {
    return attackers_to(s, occupied()) & bb_color_[by];
}

bool Position::in_check() const {
    return is_attacked(king_square(side_to_move_), ~side_to_move_);
}

// Castling rights update table — indexed by square
// When a piece moves from or to these squares, remove the corresponding rights
static constexpr CastlingRight castling_update[NUM_SQUARES] = {
    // A1       B1               C1               D1               E1                F1               G1               H1
    CastlingRight(~WHITE_OOO), ALL_CASTLING, ALL_CASTLING, ALL_CASTLING, CastlingRight(~WHITE_CASTLE), ALL_CASTLING, ALL_CASTLING, CastlingRight(~WHITE_OO),
    // Ranks 2-7: no effect
    ALL_CASTLING, ALL_CASTLING, ALL_CASTLING, ALL_CASTLING, ALL_CASTLING, ALL_CASTLING, ALL_CASTLING, ALL_CASTLING,
    ALL_CASTLING, ALL_CASTLING, ALL_CASTLING, ALL_CASTLING, ALL_CASTLING, ALL_CASTLING, ALL_CASTLING, ALL_CASTLING,
    ALL_CASTLING, ALL_CASTLING, ALL_CASTLING, ALL_CASTLING, ALL_CASTLING, ALL_CASTLING, ALL_CASTLING, ALL_CASTLING,
    ALL_CASTLING, ALL_CASTLING, ALL_CASTLING, ALL_CASTLING, ALL_CASTLING, ALL_CASTLING, ALL_CASTLING, ALL_CASTLING,
    ALL_CASTLING, ALL_CASTLING, ALL_CASTLING, ALL_CASTLING, ALL_CASTLING, ALL_CASTLING, ALL_CASTLING, ALL_CASTLING,
    ALL_CASTLING, ALL_CASTLING, ALL_CASTLING, ALL_CASTLING, ALL_CASTLING, ALL_CASTLING, ALL_CASTLING, ALL_CASTLING,
    // A8       B8               C8               D8               E8                F8               G8               H8
    CastlingRight(~BLACK_OOO), ALL_CASTLING, ALL_CASTLING, ALL_CASTLING, CastlingRight(~BLACK_CASTLE), ALL_CASTLING, ALL_CASTLING, CastlingRight(~BLACK_OO),
};

void Position::make_move(Move m, UndoInfo& undo) {
    Square from = m.from();
    Square to = m.to();
    MoveFlag flag = m.flag();
    Color us = side_to_move_;
    Color them = ~us;

    // Save undo state
    undo.castling = castling_;
    undo.ep = ep_square_;
    undo.halfmove = halfmove_clock_;
    undo.captured = NO_PIECE_TYPE;

    // Handle captures
    if (m.is_capture()) {
        Square cap_sq = to;
        if (m.is_ep()) {
            // En passant: captured pawn is on a different square
            cap_sq = (us == WHITE) ? Square(to - 8) : Square(to + 8);
        }
        undo.captured = board_[cap_sq];
        remove_piece(cap_sq);
    }

    // Move the piece
    move_piece(from, to);

    // Handle promotions
    if (m.is_promotion()) {
        // Remove the pawn we just moved, replace with promotion piece
        remove_piece(to);
        put_piece(us, m.promo_piece(), to);
    }

    // Handle castling — move the rook
    if (flag == FLAG_KING_CASTLE) {
        Square rook_from = (us == WHITE) ? H1 : H8;
        Square rook_to   = (us == WHITE) ? F1 : F8;
        move_piece(rook_from, rook_to);
    } else if (flag == FLAG_QUEEN_CASTLE) {
        Square rook_from = (us == WHITE) ? A1 : A8;
        Square rook_to   = (us == WHITE) ? D1 : D8;
        move_piece(rook_from, rook_to);
    }

    // Update en passant square
    if (flag == FLAG_DOUBLE_PUSH) {
        ep_square_ = (us == WHITE) ? Square(from + 8) : Square(from - 8);
    } else {
        ep_square_ = NO_SQUARE;
    }

    // Update castling rights
    castling_ = castling_ & castling_update[from] & castling_update[to];

    // Update clocks
    if (board_[to] == PAWN || undo.captured != NO_PIECE_TYPE)
        halfmove_clock_ = 0;
    else
        halfmove_clock_++;

    if (us == BLACK) fullmove_number_++;

    // Flip side
    side_to_move_ = them;
}

void Position::unmake_move(Move m, const UndoInfo& undo) {
    Square from = m.from();
    Square to = m.to();
    MoveFlag flag = m.flag();
    Color them = side_to_move_;     // After make, side flipped
    Color us = ~them;               // The side that made the move

    // Flip side back
    side_to_move_ = us;

    // Undo castling rook move
    if (flag == FLAG_KING_CASTLE) {
        Square rook_from = (us == WHITE) ? H1 : H8;
        Square rook_to   = (us == WHITE) ? F1 : F8;
        move_piece(rook_to, rook_from);
    } else if (flag == FLAG_QUEEN_CASTLE) {
        Square rook_from = (us == WHITE) ? A1 : A8;
        Square rook_to   = (us == WHITE) ? D1 : D8;
        move_piece(rook_to, rook_from);
    }

    // Undo promotion
    if (m.is_promotion()) {
        remove_piece(to);
        put_piece(us, PAWN, to);
    }

    // Move piece back
    move_piece(to, from);

    // Restore captured piece
    if (undo.captured != NO_PIECE_TYPE) {
        Square cap_sq = to;
        if (m.is_ep()) {
            cap_sq = (us == WHITE) ? Square(to - 8) : Square(to + 8);
        }
        put_piece(them, undo.captured, cap_sq);
    }

    // Restore state
    castling_ = undo.castling;
    ep_square_ = undo.ep;
    halfmove_clock_ = undo.halfmove;
    if (us == BLACK) fullmove_number_--;
}
```

- [ ] **Step 2c: Add Move::to_uci() to types.h**

Add at the bottom of `types.h`, or in a new `types.cpp`:

```cpp
// Append to src/core/types.h, replace the declaration with inline definition:
inline std::string Move::to_uci() const {
    std::string s;
    s += char('a' + file_of(from()));
    s += char('1' + rank_of(from()));
    s += char('a' + file_of(to()));
    s += char('1' + rank_of(to()));
    if (is_promotion()) {
        const char promo[] = "nbrq";
        s += promo[flag() & 3];
    }
    return s;
}
```

- [ ] **Step 2d: Build and run tests**

```bash
cmake --build build --config Release
ctest --test-dir build --build-config Release --output-on-failure -R Position
```

Expected: All Position tests pass.

- [ ] **Step 2e: Commit**

```bash
git add src/core/position.h src/core/position.cpp src/core/types.h tests/test_position.cpp
git commit -m "feat: Position class — FEN parsing, make/unmake, attack detection"
```

---

## Task 5: Pseudo-Legal Move Generation

**Files:**
- Modify: `src/core/movegen.h`
- Modify: `src/core/movegen.cpp`
- Replace: `tests/test_movegen.cpp`

### Step 1: Write move generation tests

- [ ] **Step 1a: Write test_movegen.cpp**

```cpp
// tests/test_movegen.cpp
#include <gtest/gtest.h>
#include "core/movegen.h"
#include "core/attacks.h"
#include <algorithm>

class MoveGenTest : public ::testing::Test {
protected:
    void SetUp() override {
        attacks::init();
    }

    // Helper: count legal moves for a position
    int count_legal_moves(const std::string& fen) {
        Position pos;
        pos.set_fen(fen);
        Move moves[MAX_MOVES];
        int n = generate_legal_moves(pos, moves);
        return n;
    }

    // Helper: check if a specific UCI move is in the legal move list
    bool has_move(const std::string& fen, const std::string& uci) {
        Position pos;
        pos.set_fen(fen);
        Move moves[MAX_MOVES];
        int n = generate_legal_moves(pos, moves);
        for (int i = 0; i < n; ++i) {
            if (moves[i].to_uci() == uci) return true;
        }
        return false;
    }
};

TEST_F(MoveGenTest, StartingPosition) {
    // 20 legal moves: 16 pawn moves + 4 knight moves
    EXPECT_EQ(count_legal_moves(
        "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"), 20);
}

TEST_F(MoveGenTest, PawnDoublePush) {
    EXPECT_TRUE(has_move(
        "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1", "e2e4"));
}

TEST_F(MoveGenTest, PawnCapture) {
    // White pawn on E4, black pawn on D5
    EXPECT_TRUE(has_move(
        "8/8/8/3p4/4P3/8/8/4K2k w - - 0 1", "e4d5"));
}

TEST_F(MoveGenTest, EnPassant) {
    // After 1. e4 d5 2. e5 f5, white can play exf6 e.p.
    EXPECT_TRUE(has_move(
        "rnbqkbnr/ppppp1pp/8/4Pp2/8/8/PPPP1PPP/RNBQKBNR w KQkq f6 0 3", "e5f6"));
}

TEST_F(MoveGenTest, Promotion) {
    // White pawn on A7, no blockers
    EXPECT_TRUE(has_move("8/P7/8/8/8/8/8/4K2k w - - 0 1", "a7a8q"));
    EXPECT_TRUE(has_move("8/P7/8/8/8/8/8/4K2k w - - 0 1", "a7a8r"));
    EXPECT_TRUE(has_move("8/P7/8/8/8/8/8/4K2k w - - 0 1", "a7a8b"));
    EXPECT_TRUE(has_move("8/P7/8/8/8/8/8/4K2k w - - 0 1", "a7a8n"));
}

TEST_F(MoveGenTest, CastlingKingside) {
    EXPECT_TRUE(has_move(
        "r3k2r/pppppppp/8/8/8/8/PPPPPPPP/R3K2R w KQkq - 0 1", "e1g1"));
}

TEST_F(MoveGenTest, CastlingQueenside) {
    EXPECT_TRUE(has_move(
        "r3k2r/pppppppp/8/8/8/8/PPPPPPPP/R3K2R w KQkq - 0 1", "e1c1"));
}

TEST_F(MoveGenTest, CastlingBlockedByPiece) {
    // Bishop on F1 blocks kingside castling
    EXPECT_FALSE(has_move(
        "r3k2r/pppppppp/8/8/8/8/PPPPPPPP/R3KB1R w KQkq - 0 1", "e1g1"));
}

TEST_F(MoveGenTest, CastlingThroughCheck) {
    // Enemy rook on F8 attacks F1, can't castle kingside
    EXPECT_FALSE(has_move(
        "5r2/8/8/8/8/8/8/R3K2R w KQ - 0 1", "e1g1"));
}

TEST_F(MoveGenTest, CastlingOutOfCheck) {
    // King in check, can't castle
    EXPECT_FALSE(has_move(
        "4r3/8/8/8/8/8/8/R3K2R w KQ - 0 1", "e1g1"));
}

TEST_F(MoveGenTest, MustEscapeCheck) {
    // Black king in check from white rook on E1. Only legal moves escape check.
    // K on E8, R on E1 gives check. Black must move king.
    int n = count_legal_moves("4k3/8/8/8/8/8/8/4R2K b - - 0 1");
    EXPECT_GT(n, 0);

    // All moves should be king moves
    Position pos;
    pos.set_fen("4k3/8/8/8/8/8/8/4R2K b - - 0 1");
    Move moves[MAX_MOVES];
    int count = generate_legal_moves(pos, moves);
    for (int i = 0; i < count; ++i) {
        EXPECT_EQ(pos.piece_on(moves[i].from()), KING);
    }
}

TEST_F(MoveGenTest, StalemateNoMoves) {
    // Black king on A8, white queen on B6, white king on C8 — stalemate
    EXPECT_EQ(count_legal_moves("k7/8/1Q6/8/8/8/8/2K5 b - - 0 1"), 0);
}

TEST_F(MoveGenTest, PinnedPieceCannotMove) {
    // White bishop on D2 pinned by black rook on A5 (through king on E1)
    // Actually let's use a simpler pin: rook pins knight
    // White king E1, white knight E2, black rook E8 — knight is pinned
    EXPECT_FALSE(has_move("4r3/8/8/8/8/8/4N3/4K3 w - - 0 1", "e2d4"));
    EXPECT_FALSE(has_move("4r3/8/8/8/8/8/4N3/4K3 w - - 0 1", "e2f4"));
}
```

- [ ] **Step 1b: Run tests — verify they fail**

```bash
cmake --build build --config Release 2>&1
```

Expected: Compile error — `generate_legal_moves` not defined.

### Step 2: Implement move generation

- [ ] **Step 2a: Write movegen.h**

```cpp
// src/core/movegen.h
#pragma once
#include "core/position.h"

// Generates all legal moves for the position.
// Returns the number of moves written to the array.
// Array must have space for MAX_MOVES entries.
int generate_legal_moves(const Position& pos, Move* moves);
```

- [ ] **Step 2b: Write movegen.cpp**

```cpp
// src/core/movegen.cpp
#include "core/movegen.h"
#include "core/attacks.h"

// Helper: add a move to the list
static inline void add_move(Move*& list, Square from, Square to, MoveFlag flag) {
    *list++ = Move(from, to, flag);
}

// Helper: add promotion moves (4 promotions, or 4 capture-promotions)
static inline void add_promotions(Move*& list, Square from, Square to, bool capture) {
    if (capture) {
        add_move(list, from, to, FLAG_PROMO_CAP_Q);
        add_move(list, from, to, FLAG_PROMO_CAP_R);
        add_move(list, from, to, FLAG_PROMO_CAP_B);
        add_move(list, from, to, FLAG_PROMO_CAP_N);
    } else {
        add_move(list, from, to, FLAG_PROMO_QUEEN);
        add_move(list, from, to, FLAG_PROMO_ROOK);
        add_move(list, from, to, FLAG_PROMO_BISHOP);
        add_move(list, from, to, FLAG_PROMO_KNIGHT);
    }
}

// Generate all pseudo-legal moves (may leave king in check)
static int generate_pseudo_legal(const Position& pos, Move* moves) {
    Move* list = moves;
    Color us = pos.side_to_move();
    Color them = ~us;
    Bitboard our_pieces = pos.occupied(us);
    Bitboard their_pieces = pos.occupied(them);
    Bitboard occ = pos.occupied();
    Bitboard empty = ~occ;

    int push_dir = (us == WHITE) ? 8 : -8;
    int promo_rank = (us == WHITE) ? 6 : 1;   // rank index where pawns promote from
    int start_rank = (us == WHITE) ? 1 : 6;   // rank index for double push

    // --- PAWNS ---
    Bitboard pawns = pos.pieces(us, PAWN);
    while (pawns) {
        Square from = pop_lsb(pawns);
        int r = rank_of(from);

        // Single push
        Square to_sq = Square(int(from) + push_dir);
        if (!(occ & square_bb(to_sq))) {
            if (r == promo_rank) {
                add_promotions(list, from, to_sq, false);
            } else {
                add_move(list, from, to_sq, FLAG_QUIET);

                // Double push
                if (r == start_rank) {
                    Square to_sq2 = Square(int(to_sq) + push_dir);
                    if (!(occ & square_bb(to_sq2))) {
                        add_move(list, from, to_sq2, FLAG_DOUBLE_PUSH);
                    }
                }
            }
        }

        // Captures
        Bitboard atk = attacks::pawn(us, from) & their_pieces;
        while (atk) {
            Square to = pop_lsb(atk);
            if (r == promo_rank) {
                add_promotions(list, from, to, true);
            } else {
                add_move(list, from, to, FLAG_CAPTURE);
            }
        }

        // En passant
        if (pos.ep_square() != NO_SQUARE) {
            if (attacks::pawn(us, from) & square_bb(pos.ep_square())) {
                add_move(list, from, pos.ep_square(), FLAG_EP_CAPTURE);
            }
        }
    }

    // --- KNIGHTS ---
    Bitboard knights = pos.pieces(us, KNIGHT);
    while (knights) {
        Square from = pop_lsb(knights);
        Bitboard atk = attacks::knight(from) & ~our_pieces;
        while (atk) {
            Square to = pop_lsb(atk);
            MoveFlag flag = (their_pieces & square_bb(to)) ? FLAG_CAPTURE : FLAG_QUIET;
            add_move(list, from, to, flag);
        }
    }

    // --- BISHOPS ---
    Bitboard bishops = pos.pieces(us, BISHOP);
    while (bishops) {
        Square from = pop_lsb(bishops);
        Bitboard atk = attacks::bishop(from, occ) & ~our_pieces;
        while (atk) {
            Square to = pop_lsb(atk);
            MoveFlag flag = (their_pieces & square_bb(to)) ? FLAG_CAPTURE : FLAG_QUIET;
            add_move(list, from, to, flag);
        }
    }

    // --- ROOKS ---
    Bitboard rooks = pos.pieces(us, ROOK);
    while (rooks) {
        Square from = pop_lsb(rooks);
        Bitboard atk = attacks::rook(from, occ) & ~our_pieces;
        while (atk) {
            Square to = pop_lsb(atk);
            MoveFlag flag = (their_pieces & square_bb(to)) ? FLAG_CAPTURE : FLAG_QUIET;
            add_move(list, from, to, flag);
        }
    }

    // --- QUEENS ---
    Bitboard queens = pos.pieces(us, QUEEN);
    while (queens) {
        Square from = pop_lsb(queens);
        Bitboard atk = attacks::queen(from, occ) & ~our_pieces;
        while (atk) {
            Square to = pop_lsb(atk);
            MoveFlag flag = (their_pieces & square_bb(to)) ? FLAG_CAPTURE : FLAG_QUIET;
            add_move(list, from, to, flag);
        }
    }

    // --- KING ---
    Square king_sq = pos.king_square(us);
    Bitboard king_atk = attacks::king(king_sq) & ~our_pieces;
    while (king_atk) {
        Square to = pop_lsb(king_atk);
        MoveFlag flag = (their_pieces & square_bb(to)) ? FLAG_CAPTURE : FLAG_QUIET;
        add_move(list, king_sq, to, flag);
    }

    // --- CASTLING ---
    CastlingRight rights = pos.castling_rights();

    if (us == WHITE) {
        if ((rights & WHITE_OO) && !(occ & (square_bb(F1) | square_bb(G1)))) {
            if (!pos.is_attacked(E1, them) && !pos.is_attacked(F1, them) && !pos.is_attacked(G1, them)) {
                add_move(list, E1, G1, FLAG_KING_CASTLE);
            }
        }
        if ((rights & WHITE_OOO) && !(occ & (square_bb(D1) | square_bb(C1) | square_bb(B1)))) {
            if (!pos.is_attacked(E1, them) && !pos.is_attacked(D1, them) && !pos.is_attacked(C1, them)) {
                add_move(list, E1, C1, FLAG_QUEEN_CASTLE);
            }
        }
    } else {
        if ((rights & BLACK_OO) && !(occ & (square_bb(F8) | square_bb(G8)))) {
            if (!pos.is_attacked(E8, them) && !pos.is_attacked(F8, them) && !pos.is_attacked(G8, them)) {
                add_move(list, E8, G8, FLAG_KING_CASTLE);
            }
        }
        if ((rights & BLACK_OOO) && !(occ & (square_bb(D8) | square_bb(C8) | square_bb(B8)))) {
            if (!pos.is_attacked(E8, them) && !pos.is_attacked(D8, them) && !pos.is_attacked(C8, them)) {
                add_move(list, E8, C8, FLAG_QUEEN_CASTLE);
            }
        }
    }

    return int(list - moves);
}

int generate_legal_moves(const Position& pos, Move* moves) {
    Move pseudo[MAX_MOVES];
    int n = generate_pseudo_legal(pos, pseudo);

    // Filter: only keep moves that don't leave our king in check
    int legal_count = 0;
    for (int i = 0; i < n; ++i) {
        Position tmp = pos; // Copy position
        UndoInfo undo;
        tmp.make_move(pseudo[i], undo);

        // After make_move, side_to_move is the opponent.
        // Check if the side that just moved is in check.
        Color moved_side = ~tmp.side_to_move();
        if (!tmp.is_attacked(tmp.king_square(moved_side), tmp.side_to_move())) {
            moves[legal_count++] = pseudo[i];
        }
    }

    return legal_count;
}
```

Note: This uses position copying for legality checking. It's correct but not optimal — a future optimization can use incremental check detection. For MVP correctness, this is the right approach.

- [ ] **Step 2c: Build and run tests**

```bash
cmake --build build --config Release
ctest --test-dir build --build-config Release --output-on-failure -R MoveGen
```

Expected: All MoveGen tests pass.

- [ ] **Step 2d: Commit**

```bash
git add src/core/movegen.h src/core/movegen.cpp tests/test_movegen.cpp
git commit -m "feat: full legal move generation — pawns, pieces, castling, en passant"
```

---

## Task 6: Perft Validation

**Files:**
- Modify: `src/main.cpp`
- Replace: `tests/test_perft.cpp`

Perft (performance test) counts all leaf nodes of a move tree at a given depth. This is the gold standard for move generation correctness. We compare our counts against known-correct results from the chess programming community.

### Step 1: Write perft tests

- [ ] **Step 1a: Write test_perft.cpp**

```cpp
// tests/test_perft.cpp
#include <gtest/gtest.h>
#include "core/movegen.h"
#include "core/attacks.h"

class PerftTest : public ::testing::Test {
protected:
    void SetUp() override {
        attacks::init();
    }

    uint64_t perft(Position& pos, int depth) {
        if (depth == 0) return 1;

        Move moves[MAX_MOVES];
        int n = generate_legal_moves(pos, moves);

        if (depth == 1) return n; // Leaf node optimization

        uint64_t nodes = 0;
        for (int i = 0; i < n; ++i) {
            UndoInfo undo;
            pos.make_move(moves[i], undo);
            nodes += perft(pos, depth - 1);
            pos.unmake_move(moves[i], undo);
        }
        return nodes;
    }
};

// Position 1: Starting position
// https://www.chessprogramming.org/Perft_Results
TEST_F(PerftTest, StartingPos_Depth1) {
    Position pos;
    pos.set_fen("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1");
    EXPECT_EQ(perft(pos, 1), 20ULL);
}

TEST_F(PerftTest, StartingPos_Depth2) {
    Position pos;
    pos.set_fen("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1");
    EXPECT_EQ(perft(pos, 2), 400ULL);
}

TEST_F(PerftTest, StartingPos_Depth3) {
    Position pos;
    pos.set_fen("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1");
    EXPECT_EQ(perft(pos, 3), 8902ULL);
}

TEST_F(PerftTest, StartingPos_Depth4) {
    Position pos;
    pos.set_fen("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1");
    EXPECT_EQ(perft(pos, 4), 197281ULL);
}

TEST_F(PerftTest, StartingPos_Depth5) {
    Position pos;
    pos.set_fen("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1");
    EXPECT_EQ(perft(pos, 5), 4865609ULL);
}

// Position 2: "Kiwipete" — stress test for pins, en passant, castling
// https://www.chessprogramming.org/Perft_Results#Position_2
TEST_F(PerftTest, Kiwipete_Depth1) {
    Position pos;
    pos.set_fen("r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1");
    EXPECT_EQ(perft(pos, 1), 48ULL);
}

TEST_F(PerftTest, Kiwipete_Depth2) {
    Position pos;
    pos.set_fen("r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1");
    EXPECT_EQ(perft(pos, 2), 2039ULL);
}

TEST_F(PerftTest, Kiwipete_Depth3) {
    Position pos;
    pos.set_fen("r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1");
    EXPECT_EQ(perft(pos, 3), 97862ULL);
}

TEST_F(PerftTest, Kiwipete_Depth4) {
    Position pos;
    pos.set_fen("r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1");
    EXPECT_EQ(perft(pos, 4), 4085603ULL);
}

// Position 3: en passant and promotion edge cases
TEST_F(PerftTest, Position3_Depth1) {
    Position pos;
    pos.set_fen("8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1");
    EXPECT_EQ(perft(pos, 1), 14ULL);
}

TEST_F(PerftTest, Position3_Depth2) {
    Position pos;
    pos.set_fen("8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1");
    EXPECT_EQ(perft(pos, 2), 191ULL);
}

TEST_F(PerftTest, Position3_Depth3) {
    Position pos;
    pos.set_fen("8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1");
    EXPECT_EQ(perft(pos, 3), 2812ULL);
}

TEST_F(PerftTest, Position3_Depth4) {
    Position pos;
    pos.set_fen("8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1");
    EXPECT_EQ(perft(pos, 4), 43238ULL);
}

TEST_F(PerftTest, Position3_Depth5) {
    Position pos;
    pos.set_fen("8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1");
    EXPECT_EQ(perft(pos, 5), 674624ULL);
}

// Position 4: mirrored castling
TEST_F(PerftTest, Position4_Depth1) {
    Position pos;
    pos.set_fen("r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq - 0 1");
    EXPECT_EQ(perft(pos, 1), 6ULL);
}

TEST_F(PerftTest, Position4_Depth2) {
    Position pos;
    pos.set_fen("r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq - 0 1");
    EXPECT_EQ(perft(pos, 2), 264ULL);
}

TEST_F(PerftTest, Position4_Depth3) {
    Position pos;
    pos.set_fen("r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq - 0 1");
    EXPECT_EQ(perft(pos, 3), 9467ULL);
}

TEST_F(PerftTest, Position4_Depth4) {
    Position pos;
    pos.set_fen("r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq - 0 1");
    EXPECT_EQ(perft(pos, 4), 422333ULL);
}

// Position 5: complex midgame
TEST_F(PerftTest, Position5_Depth1) {
    Position pos;
    pos.set_fen("rnbq1k1r/pp1Pbppp/2p5/8/2B5/8/PPP1NnPP/RNBQK2R w KQ - 1 8");
    EXPECT_EQ(perft(pos, 1), 44ULL);
}

TEST_F(PerftTest, Position5_Depth2) {
    Position pos;
    pos.set_fen("rnbq1k1r/pp1Pbppp/2p5/8/2B5/8/PPP1NnPP/RNBQK2R w KQ - 1 8");
    EXPECT_EQ(perft(pos, 2), 1486ULL);
}

TEST_F(PerftTest, Position5_Depth3) {
    Position pos;
    pos.set_fen("rnbq1k1r/pp1Pbppp/2p5/8/2B5/8/PPP1NnPP/RNBQK2R w KQ - 1 8");
    EXPECT_EQ(perft(pos, 3), 62379ULL);
}

TEST_F(PerftTest, Position5_Depth4) {
    Position pos;
    pos.set_fen("rnbq1k1r/pp1Pbppp/2p5/8/2B5/8/PPP1NnPP/RNBQK2R w KQ - 1 8");
    EXPECT_EQ(perft(pos, 4), 2103487ULL);
}
```

- [ ] **Step 1b: Run tests — verify they fail (or some fail)**

```bash
cmake --build build --config Release
ctest --test-dir build --build-config Release --output-on-failure -R Perft
```

If all perft tests pass on first try, congratulations — your move generation is correct. More likely, some will fail and you'll need to debug. See debugging section below.

### Step 2: Perft debugging (if tests fail)

- [ ] **Step 2a: Add divide perft to main.cpp for debugging**

```cpp
// src/main.cpp
#include "core/movegen.h"
#include "core/attacks.h"
#include <iostream>
#include <string>

uint64_t perft(Position& pos, int depth) {
    if (depth == 0) return 1;

    Move moves[MAX_MOVES];
    int n = generate_legal_moves(pos, moves);

    if (depth == 1) return n;

    uint64_t nodes = 0;
    for (int i = 0; i < n; ++i) {
        UndoInfo undo;
        pos.make_move(moves[i], undo);
        nodes += perft(pos, depth - 1);
        pos.unmake_move(moves[i], undo);
    }
    return nodes;
}

// "Divide" perft: shows node count per first move (for comparing against known-good engines)
void divide(Position& pos, int depth) {
    Move moves[MAX_MOVES];
    int n = generate_legal_moves(pos, moves);

    uint64_t total = 0;
    for (int i = 0; i < n; ++i) {
        UndoInfo undo;
        pos.make_move(moves[i], undo);
        uint64_t nodes = (depth <= 1) ? 1 : perft(pos, depth - 1);
        pos.unmake_move(moves[i], undo);
        std::cout << moves[i].to_uci() << ": " << nodes << "\n";
        total += nodes;
    }
    std::cout << "\nTotal: " << total << "\n";
    std::cout << "Moves: " << n << "\n";
}

int main(int argc, char* argv[]) {
    attacks::init();

    std::string fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";
    int depth = 5;

    if (argc >= 2) depth = std::stoi(argv[1]);
    if (argc >= 3) {
        fen = "";
        for (int i = 2; i < argc; ++i) {
            if (i > 2) fen += " ";
            fen += argv[i];
        }
    }

    Position pos;
    pos.set_fen(fen);

    std::cout << "FEN: " << pos.to_fen() << "\n";
    std::cout << "Depth: " << depth << "\n\n";

    divide(pos, depth);
    return 0;
}
```

When a perft result is wrong, compare the `divide` output of your engine against a known-correct engine (e.g., Stockfish's `go perft N` command, or an online perft calculator). The first move with a different node count points to the bug.

- [ ] **Step 2b: Build and test the perft CLI**

```bash
cmake --build build --config Release
./build/Release/chess_engine.exe 3
```

Expected output (starting position depth 3):
```
FEN: rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1
Depth: 3

a2a3: 380
b2b3: 420
c2c3: 420
d2d3: 539
e2e3: 599
f2f3: 380
g2g3: 420
h2h3: 380
a2a4: 420
b2b4: 421
c2c4: 441
d2d4: 560
e2e4: 600
f2f4: 401
g2g4: 421
h2h4: 420
b1a3: 400
b1c3: 440
g1f3: 440
g1h3: 400

Total: 8902
Moves: 20
```

- [ ] **Step 2c: Iterate until all perft tests pass**

Debug strategy:
1. Run perft tests. Note which position/depth fails.
2. Run divide at the failing depth.
3. Compare against Stockfish or https://www.chessprogramming.org/Perft_Results
4. Find the first move with wrong count.
5. Set up that position after the move. Run divide again at depth-1.
6. Repeat until you find the leaf position where the move count is wrong.
7. Fix the bug in movegen or make/unmake.

Common bugs:
- En passant not clearing properly after make/unmake
- Castling rights not updated when rook is captured (not just moved)
- Pawn promotion captures generating wrong flags
- King walking into check not filtered (should be handled by legality filter)
- En passant discovered check (rare but position 3 tests it)

- [ ] **Step 2d: Commit when all perft tests pass**

```bash
git add src/main.cpp tests/test_perft.cpp
git commit -m "feat: perft validation — all standard test positions pass"
```

---

## Task 7: Final Integration Test & Cleanup

**Files:**
- All source files (review)

### Step 1: Run full test suite

- [ ] **Step 1a: Clean build and full test run**

```bash
cd E:/dev/chess-ai
rm -rf build
cmake -B build -G "Visual Studio 17 2022"
cmake --build build --config Release
ctest --test-dir build --build-config Release --output-on-failure
```

Expected: ALL tests pass (Types, Bitboard, Attacks, Position, MoveGen, Perft).

- [ ] **Step 1b: Run perft depth 5 from CLI to confirm speed**

```bash
time ./build/Release/chess_engine.exe 5
```

Expected: Starting position depth 5 = 4,865,609 nodes. On a Ryzen 7 5800X this should complete in under 5 seconds (without optimization). If it's much slower, something is wrong.

### Step 2: Commit final state

- [ ] **Step 2a: Final commit**

```bash
git add -A
git commit -m "milestone: chess engine core complete — all perft tests pass"
```

---

## Summary

After completing this plan, you will have:

- **Board representation**: Bitboard-based position with FEN I/O
- **Attack tables**: Precomputed knight/king/pawn, ray-based bishop/rook/queen
- **Legal move generation**: All move types including castling, en passant, promotions
- **Make/unmake**: Full position state save/restore
- **Validation**: Perft tests against 5 standard positions at multiple depths
- **CLI tool**: Perft divide for debugging

**What's NOT included (saved for later plans):**
- Zobrist hashing (Plan 2: MCTS needs it for transposition detection)
- Move ordering (Plan 2: MCTS)
- Magic bitboards (optimization, can be added in Plan 2 or later)
- Neural network anything (Plans 3-5)
- Visualization (Plan 6)

## Next Plans

| Plan | Ready to Start | Key Question |
|------|----------------|--------------|
| Plan 2: MCTS | After this plan passes perft | Random rollouts or NN-guided from start? |
| Plan 3: NN Training | Can start in parallel NOW | Network size (residual blocks)? |
| Plan 4: Self-Play | After Plans 1 + 2 | Data format (protobuf vs custom binary)? |
| Plan 5: Integration | After Plans 1-4 | TorchScript vs ONNX Runtime for C++ inference? |
| Plan 6: Visualization | After Plan 5 MVP | WebSocket vs polling? |
