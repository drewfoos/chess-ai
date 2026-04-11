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
