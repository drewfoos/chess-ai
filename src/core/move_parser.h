#pragma once
#include "core/types.h"
#include "core/position.h"
#include "core/movegen.h"
#include <string>
#include <stdexcept>

inline Move parse_uci_move(const Position& pos, const std::string& uci) {
    if (uci.size() < 4 || uci.size() > 5) {
        throw std::runtime_error("Invalid UCI move format: " + uci);
    }
    int from = (uci[0] - 'a') + (uci[1] - '1') * 8;
    int to   = (uci[2] - 'a') + (uci[3] - '1') * 8;
    PieceType promo = NO_PIECE_TYPE;
    if (uci.size() == 5) {
        switch (uci[4]) {
            case 'q': promo = QUEEN;  break;
            case 'r': promo = ROOK;   break;
            case 'b': promo = BISHOP; break;
            case 'n': promo = KNIGHT; break;
            default: throw std::runtime_error("Invalid promotion piece in UCI move: " + uci);
        }
    }
    Move moves[MAX_MOVES];
    int n = generate_legal_moves(pos, moves);
    for (int i = 0; i < n; i++) {
        if (moves[i].from() == Square(from) && moves[i].to() == Square(to)) {
            if (promo == NO_PIECE_TYPE || moves[i].promo_piece() == promo) {
                return moves[i];
            }
        }
    }
    throw std::runtime_error("Illegal move: " + uci);
}
