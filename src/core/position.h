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

    void set_fen(const std::string& fen);
    std::string to_fen() const;

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

    bool is_attacked(Square s, Color by) const;
    bool in_check() const;
    Bitboard attackers_to(Square s, Bitboard occ) const;

    void make_move(Move m, UndoInfo& undo);
    void unmake_move(Move m, const UndoInfo& undo);

private:
    Bitboard bb_pieces_[NUM_COLORS][NUM_PIECE_TYPES] = {};
    Bitboard bb_color_[NUM_COLORS] = {};
    PieceType board_[NUM_SQUARES] = {};
    Color color_[NUM_SQUARES] = {};

    Color side_to_move_ = WHITE;
    CastlingRight castling_ = NO_CASTLING;
    Square ep_square_ = NO_SQUARE;
    int halfmove_clock_ = 0;
    int fullmove_number_ = 1;

    void put_piece(Color c, PieceType pt, Square s);
    void remove_piece(Square s);
    void move_piece(Square from, Square to);
};
