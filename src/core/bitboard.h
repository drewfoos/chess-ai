#pragma once
#include "core/types.h"
#include <string>

#ifdef _MSC_VER
#include <intrin.h>
#endif

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

constexpr Bitboard square_bb(Square s) { return 1ULL << s; }

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

std::string bitboard_to_string(Bitboard bb);
