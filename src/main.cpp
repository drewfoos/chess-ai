#include "core/types.h"
#include "core/bitboard.h"
#include "core/attacks.h"
#include "core/position.h"
#include "core/movegen.h"
#include "mcts/search.h"
#include <iostream>
#include <string>
#include <cstdlib>
#include <iomanip>
#include <algorithm>

static uint64_t perft(Position& pos, int depth) {
    if (depth == 0) return 1;
    Move moves[MAX_MOVES];
    int n = generate_legal_moves(pos, moves);
    if (depth == 1) return n;
    uint64_t nodes = 0;
    UndoInfo undo;
    for (int i = 0; i < n; i++) {
        pos.make_move(moves[i], undo);
        nodes += perft(pos, depth - 1);
        pos.unmake_move(moves[i], undo);
    }
    return nodes;
}

static void divide(Position& pos, int depth) {
    Move moves[MAX_MOVES];
    int n = generate_legal_moves(pos, moves);
    uint64_t total = 0;
    UndoInfo undo;
    for (int i = 0; i < n; i++) {
        pos.make_move(moves[i], undo);
        uint64_t count = (depth <= 1) ? 1 : perft(pos, depth - 1);
        pos.unmake_move(moves[i], undo);
        std::cout << moves[i].to_uci() << ": " << count << "\n";
        total += count;
    }
    std::cout << "\nTotal: " << total << "\n";
}

static void search_position(const std::string& fen, int iterations) {
    Position pos;
    pos.set_fen(fen);

    mcts::RandomEvaluator eval;
    mcts::SearchParams params;
    params.num_iterations = iterations;
    params.add_noise = false;

    mcts::Search search(eval, params);
    mcts::SearchResult result = search.run(pos);

    if (result.best_move.is_none()) {
        std::cout << "No legal moves (";
        if (pos.in_check()) std::cout << "checkmate";
        else std::cout << "stalemate";
        std::cout << ")\n";
        return;
    }

    std::cout << "Position: " << pos.to_fen() << "\n";
    std::cout << "Iterations: " << iterations << "\n";
    std::cout << "Root value: " << std::fixed << std::setprecision(3) << result.root_value << "\n";
    std::cout << "Best move: " << result.best_move.to_uci() << "\n\n";

    // Sort by visit count descending for display
    std::vector<std::pair<int, int>> indexed(result.moves.size());
    for (int i = 0; i < static_cast<int>(result.moves.size()); i++) {
        indexed[i] = {result.visit_counts[i], i};
    }
    std::sort(indexed.begin(), indexed.end(), std::greater<>());

    int total_visits = 0;
    for (int v : result.visit_counts) total_visits += v;

    std::cout << "Move distribution (top 10):\n";
    int shown = 0;
    for (auto& [visits, idx] : indexed) {
        if (shown >= 10) break;
        float pct = 100.0f * visits / total_visits;
        std::cout << "  " << std::setw(5) << result.moves[idx].to_uci()
                  << "  " << std::setw(6) << visits << " visits"
                  << "  (" << std::fixed << std::setprecision(1) << pct << "%)\n";
        shown++;
    }
}

int main(int argc, char* argv[]) {
    attacks::init();

    if (argc < 2) {
        std::cout << "Usage:\n";
        std::cout << "  chess_engine perft <depth> [fen]\n";
        std::cout << "  chess_engine search [fen] [iterations]\n";
        return 1;
    }

    std::string command = argv[1];

    if (command == "perft") {
        if (argc < 3) {
            std::cerr << "Usage: chess_engine perft <depth> [fen]\n";
            return 1;
        }
        int depth = std::atoi(argv[2]);
        std::string fen = (argc >= 4) ? argv[3]
            : "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";
        Position pos;
        pos.set_fen(fen);
        divide(pos, depth);
    } else if (command == "search") {
        std::string fen = (argc >= 3) ? argv[2]
            : "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";
        int iterations = (argc >= 4) ? std::atoi(argv[3]) : 800;
        search_position(fen, iterations);
    } else {
        std::cerr << "Unknown command: " << command << "\n";
        return 1;
    }

    return 0;
}
