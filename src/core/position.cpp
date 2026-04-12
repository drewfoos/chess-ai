#include "core/position.h"
#include "core/attacks.h"
#include <sstream>
#include <algorithm>

// Castling rights update table: castling_ &= castle_mask[from] & castle_mask[to]
static CastlingRight castle_mask[NUM_SQUARES];

static bool castle_mask_init = []() {
    for (int i = 0; i < NUM_SQUARES; ++i)
        castle_mask[i] = ALL_CASTLING;
    castle_mask[A1] = CastlingRight(ALL_CASTLING & ~WHITE_OOO);
    castle_mask[E1] = CastlingRight(ALL_CASTLING & ~WHITE_CASTLE);
    castle_mask[H1] = CastlingRight(ALL_CASTLING & ~WHITE_OO);
    castle_mask[A8] = CastlingRight(ALL_CASTLING & ~BLACK_OOO);
    castle_mask[E8] = CastlingRight(ALL_CASTLING & ~BLACK_CASTLE);
    castle_mask[H8] = CastlingRight(ALL_CASTLING & ~BLACK_OO);
    return true;
}();

static const char* piece_chars = "pnbrqk";

static PieceType char_to_pt(char c) {
    c = std::tolower(c);
    for (int i = 0; i < 6; ++i)
        if (piece_chars[i] == c) return PieceType(i);
    return NO_PIECE_TYPE;
}

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
    // Clear
    for (int c = 0; c < NUM_COLORS; ++c) {
        bb_color_[c] = 0;
        for (int pt = 0; pt < NUM_PIECE_TYPES; ++pt)
            bb_pieces_[c][pt] = 0;
    }
    for (int s = 0; s < NUM_SQUARES; ++s) {
        board_[s] = NO_PIECE_TYPE;
        color_[s] = WHITE;
    }
    ep_square_ = NO_SQUARE;
    castling_ = NO_CASTLING;
    halfmove_clock_ = 0;
    fullmove_number_ = 1;

    std::istringstream ss(fen);
    std::string pieces, side, castling, ep;
    int hmc = 0, fmn = 1;
    ss >> pieces >> side >> castling >> ep;
    if (ss) ss >> hmc;
    if (ss) ss >> fmn;

    // Parse pieces
    int rank = 7, file = 0;
    for (char ch : pieces) {
        if (ch == '/') {
            rank--;
            file = 0;
        } else if (ch >= '1' && ch <= '8') {
            file += ch - '0';
        } else {
            Color c = std::isupper(ch) ? WHITE : BLACK;
            PieceType pt = char_to_pt(ch);
            put_piece(c, pt, make_square(file, rank));
            file++;
        }
    }

    side_to_move_ = (side == "b") ? BLACK : WHITE;

    if (castling != "-") {
        for (char ch : castling) {
            if (ch == 'K') castling_ = castling_ | WHITE_OO;
            else if (ch == 'Q') castling_ = castling_ | WHITE_OOO;
            else if (ch == 'k') castling_ = castling_ | BLACK_OO;
            else if (ch == 'q') castling_ = castling_ | BLACK_OOO;
        }
    }

    if (ep != "-" && ep.size() == 2) {
        int f = ep[0] - 'a';
        int r = ep[1] - '1';
        ep_square_ = make_square(f, r);
    }

    halfmove_clock_ = hmc;
    fullmove_number_ = fmn;
}

std::string Position::to_fen() const {
    std::string fen;

    // Pieces
    for (int rank = 7; rank >= 0; --rank) {
        int empty = 0;
        for (int file = 0; file < 8; ++file) {
            Square s = make_square(file, rank);
            PieceType pt = board_[s];
            if (pt == NO_PIECE_TYPE) {
                empty++;
            } else {
                if (empty > 0) {
                    fen += char('0' + empty);
                    empty = 0;
                }
                char c = piece_chars[pt];
                if (color_[s] == WHITE) c = std::toupper(c);
                fen += c;
            }
        }
        if (empty > 0) fen += char('0' + empty);
        if (rank > 0) fen += '/';
    }

    fen += (side_to_move_ == WHITE) ? " w " : " b ";

    // Castling
    if (castling_ == NO_CASTLING) {
        fen += '-';
    } else {
        if (castling_ & WHITE_OO)  fen += 'K';
        if (castling_ & WHITE_OOO) fen += 'Q';
        if (castling_ & BLACK_OO)  fen += 'k';
        if (castling_ & BLACK_OOO) fen += 'q';
    }

    fen += ' ';

    // En passant
    if (ep_square_ == NO_SQUARE) {
        fen += '-';
    } else {
        fen += char('a' + file_of(ep_square_));
        fen += char('1' + rank_of(ep_square_));
    }

    fen += ' ';
    fen += std::to_string(halfmove_clock_);
    fen += ' ';
    fen += std::to_string(fullmove_number_);

    return fen;
}

Bitboard Position::attackers_to(Square s, Bitboard occ) const {
    return (attacks::pawn(BLACK, s) & bb_pieces_[WHITE][PAWN])
         | (attacks::pawn(WHITE, s) & bb_pieces_[BLACK][PAWN])
         | (attacks::knight(s)      & (bb_pieces_[WHITE][KNIGHT] | bb_pieces_[BLACK][KNIGHT]))
         | (attacks::bishop(s, occ) & (bb_pieces_[WHITE][BISHOP] | bb_pieces_[BLACK][BISHOP]
                                      | bb_pieces_[WHITE][QUEEN]  | bb_pieces_[BLACK][QUEEN]))
         | (attacks::rook(s, occ)   & (bb_pieces_[WHITE][ROOK]   | bb_pieces_[BLACK][ROOK]
                                      | bb_pieces_[WHITE][QUEEN]  | bb_pieces_[BLACK][QUEEN]))
         | (attacks::king(s)        & (bb_pieces_[WHITE][KING]    | bb_pieces_[BLACK][KING]));
}

bool Position::is_attacked(Square s, Color by) const {
    return (attackers_to(s, occupied()) & bb_color_[by]) != 0;
}

bool Position::in_check() const {
    return is_attacked(king_square(side_to_move_), ~side_to_move_);
}

void Position::make_move(Move m, UndoInfo& undo) {
    // Save undo state
    undo.castling = castling_;
    undo.ep = ep_square_;
    undo.halfmove = halfmove_clock_;
    undo.captured = NO_PIECE_TYPE;

    Square from = m.from();
    Square to   = m.to();
    MoveFlag flag = m.flag();
    Color us = side_to_move_;
    Color them = ~us;

    // Handle captures
    if (m.is_capture()) {
        if (m.is_ep()) {
            // The captured pawn is on a different square
            Square cap_sq = (us == WHITE) ? Square(to - 8) : Square(to + 8);
            undo.captured = PAWN;
            remove_piece(cap_sq);
        } else {
            undo.captured = board_[to];
            remove_piece(to);
        }
    }

    // Move the piece
    move_piece(from, to);

    // Handle promotions
    if (m.is_promotion()) {
        // Remove the pawn at destination, put promotion piece
        remove_piece(to);
        put_piece(us, m.promo_piece(), to);
    }

    // Handle castling (move the rook)
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
    castling_ = castling_ & castle_mask[from] & castle_mask[to];

    // Update clocks
    if (board_[to] == PAWN || m.is_capture() || m.is_promotion()) {
        halfmove_clock_ = 0;
    } else {
        halfmove_clock_++;
    }

    if (us == BLACK) fullmove_number_++;

    // Flip side
    side_to_move_ = them;
}

void Position::unmake_move(Move m, const UndoInfo& undo) {
    Square from = m.from();
    Square to   = m.to();
    MoveFlag flag = m.flag();

    // Flip side back
    side_to_move_ = ~side_to_move_;
    Color us = side_to_move_;

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
        if (m.is_ep()) {
            Square cap_sq = (us == WHITE) ? Square(to - 8) : Square(to + 8);
            put_piece(~us, PAWN, cap_sq);
        } else {
            put_piece(~us, undo.captured, to);
        }
    }

    // Restore saved state
    castling_ = undo.castling;
    ep_square_ = undo.ep;
    halfmove_clock_ = undo.halfmove;
    if (us == BLACK) fullmove_number_--;
}
