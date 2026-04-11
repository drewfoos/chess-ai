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

    bool is_capture()   const { return ((flag() & 4) && !(flag() & 8)) || (flag() >= FLAG_PROMO_CAP_N); }
    bool is_promotion() const { return flag() >= FLAG_PROMO_KNIGHT; }
    bool is_castle()    const { return flag() == FLAG_KING_CASTLE || flag() == FLAG_QUEEN_CASTLE; }
    bool is_ep()        const { return flag() == FLAG_EP_CAPTURE; }

    PieceType promo_piece() const { return PieceType((flag() & 3) + KNIGHT); }

    bool operator==(Move o) const { return data == o.data; }
    bool operator!=(Move o) const { return data != o.data; }

    static Move none() { return Move(); }
    bool is_none() const { return data == 0; }

    inline std::string to_uci() const {
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

enum Direction : int {
    NORTH =  8, SOUTH = -8, EAST =  1, WEST = -1,
    NORTH_EAST = 9, NORTH_WEST = 7, SOUTH_EAST = -7, SOUTH_WEST = -9
};

constexpr int MAX_MOVES = 256;
