#include "core/movegen.h"
#include "core/attacks.h"

namespace {

int generate_pseudo_legal(const Position& pos, Move* moves) {
    int count = 0;
    const Color us = pos.side_to_move();
    const Color them = ~us;
    const Bitboard our_pieces = pos.occupied(us);
    const Bitboard their_pieces = pos.occupied(them);
    const Bitboard occ = pos.occupied();

    // === PAWNS ===
    {
        const int push_dir = (us == WHITE) ? 8 : -8;
        const int start_rank = (us == WHITE) ? 1 : 6;
        const int promo_rank = (us == WHITE) ? 6 : 1;

        Bitboard pawns = pos.pieces(us, PAWN);
        while (pawns) {
            Square from = pop_lsb(pawns);
            int from_rank = rank_of(from);

            // Single push
            Square to = from + push_dir;
            if (to < NUM_SQUARES && !(occ & square_bb(to))) {
                if (from_rank == promo_rank) {
                    moves[count++] = Move(from, to, FLAG_PROMO_KNIGHT);
                    moves[count++] = Move(from, to, FLAG_PROMO_BISHOP);
                    moves[count++] = Move(from, to, FLAG_PROMO_ROOK);
                    moves[count++] = Move(from, to, FLAG_PROMO_QUEEN);
                } else {
                    moves[count++] = Move(from, to, FLAG_QUIET);

                    // Double push
                    if (from_rank == start_rank) {
                        Square to2 = to + push_dir;
                        if (!(occ & square_bb(to2))) {
                            moves[count++] = Move(from, to2, FLAG_DOUBLE_PUSH);
                        }
                    }
                }
            }

            // Captures
            Bitboard captures = attacks::pawn(us, from) & their_pieces;
            while (captures) {
                Square cap_sq = pop_lsb(captures);
                if (from_rank == promo_rank) {
                    moves[count++] = Move(from, cap_sq, FLAG_PROMO_CAP_N);
                    moves[count++] = Move(from, cap_sq, FLAG_PROMO_CAP_B);
                    moves[count++] = Move(from, cap_sq, FLAG_PROMO_CAP_R);
                    moves[count++] = Move(from, cap_sq, FLAG_PROMO_CAP_Q);
                } else {
                    moves[count++] = Move(from, cap_sq, FLAG_CAPTURE);
                }
            }

            // En passant
            if (pos.ep_square() != NO_SQUARE) {
                if (attacks::pawn(us, from) & square_bb(pos.ep_square())) {
                    moves[count++] = Move(from, pos.ep_square(), FLAG_EP_CAPTURE);
                }
            }
        }
    }

    // === KNIGHTS ===
    {
        Bitboard knights = pos.pieces(us, KNIGHT);
        while (knights) {
            Square from = pop_lsb(knights);
            Bitboard targets = attacks::knight(from) & ~our_pieces;
            while (targets) {
                Square to = pop_lsb(targets);
                MoveFlag flag = (their_pieces & square_bb(to)) ? FLAG_CAPTURE : FLAG_QUIET;
                moves[count++] = Move(from, to, flag);
            }
        }
    }

    // === BISHOPS ===
    {
        Bitboard bishops = pos.pieces(us, BISHOP);
        while (bishops) {
            Square from = pop_lsb(bishops);
            Bitboard targets = attacks::bishop(from, occ) & ~our_pieces;
            while (targets) {
                Square to = pop_lsb(targets);
                MoveFlag flag = (their_pieces & square_bb(to)) ? FLAG_CAPTURE : FLAG_QUIET;
                moves[count++] = Move(from, to, flag);
            }
        }
    }

    // === ROOKS ===
    {
        Bitboard rooks = pos.pieces(us, ROOK);
        while (rooks) {
            Square from = pop_lsb(rooks);
            Bitboard targets = attacks::rook(from, occ) & ~our_pieces;
            while (targets) {
                Square to = pop_lsb(targets);
                MoveFlag flag = (their_pieces & square_bb(to)) ? FLAG_CAPTURE : FLAG_QUIET;
                moves[count++] = Move(from, to, flag);
            }
        }
    }

    // === QUEENS ===
    {
        Bitboard queens = pos.pieces(us, QUEEN);
        while (queens) {
            Square from = pop_lsb(queens);
            Bitboard targets = attacks::queen(from, occ) & ~our_pieces;
            while (targets) {
                Square to = pop_lsb(targets);
                MoveFlag flag = (their_pieces & square_bb(to)) ? FLAG_CAPTURE : FLAG_QUIET;
                moves[count++] = Move(from, to, flag);
            }
        }
    }

    // === KING ===
    {
        Square king_sq = pos.king_square(us);
        Bitboard targets = attacks::king(king_sq) & ~our_pieces;
        while (targets) {
            Square to = pop_lsb(targets);
            MoveFlag flag = (their_pieces & square_bb(to)) ? FLAG_CAPTURE : FLAG_QUIET;
            moves[count++] = Move(king_sq, to, flag);
        }
    }

    // === CASTLING ===
    {
        CastlingRight rights = pos.castling_rights();
        Square king_sq = pos.king_square(us);

        if (us == WHITE) {
            if ((rights & WHITE_OO) &&
                !(occ & (square_bb(F1) | square_bb(G1))) &&
                !pos.is_attacked(E1, BLACK) &&
                !pos.is_attacked(F1, BLACK) &&
                !pos.is_attacked(G1, BLACK)) {
                moves[count++] = Move(E1, G1, FLAG_KING_CASTLE);
            }
            if ((rights & WHITE_OOO) &&
                !(occ & (square_bb(D1) | square_bb(C1) | square_bb(B1))) &&
                !pos.is_attacked(E1, BLACK) &&
                !pos.is_attacked(D1, BLACK) &&
                !pos.is_attacked(C1, BLACK)) {
                moves[count++] = Move(E1, C1, FLAG_QUEEN_CASTLE);
            }
        } else {
            if ((rights & BLACK_OO) &&
                !(occ & (square_bb(F8) | square_bb(G8))) &&
                !pos.is_attacked(E8, WHITE) &&
                !pos.is_attacked(F8, WHITE) &&
                !pos.is_attacked(G8, WHITE)) {
                moves[count++] = Move(E8, G8, FLAG_KING_CASTLE);
            }
            if ((rights & BLACK_OOO) &&
                !(occ & (square_bb(D8) | square_bb(C8) | square_bb(B8))) &&
                !pos.is_attacked(E8, WHITE) &&
                !pos.is_attacked(D8, WHITE) &&
                !pos.is_attacked(C8, WHITE)) {
                moves[count++] = Move(E8, C8, FLAG_QUEEN_CASTLE);
            }
        }
    }

    return count;
}

} // anonymous namespace

int generate_legal_moves(const Position& pos, Move* moves) {
    Move pseudo[MAX_MOVES];
    int n = generate_pseudo_legal(pos, pseudo);

    int legal_count = 0;
    for (int i = 0; i < n; ++i) {
        Position tmp = pos;
        UndoInfo undo;
        tmp.make_move(pseudo[i], undo);
        Color moved_side = ~tmp.side_to_move();
        if (!tmp.is_attacked(tmp.king_square(moved_side), tmp.side_to_move())) {
            moves[legal_count++] = pseudo[i];
        }
    }
    return legal_count;
}
