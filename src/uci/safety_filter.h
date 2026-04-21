#pragma once
#include "core/position.h"
#include "core/types.h"

namespace uci {

// Post-search tactical guard: rejects candidate moves that hand the opponent
// an immediate forced mate. Runs after MCTS selects a best move, before the
// engine emits `bestmove`. Exists because a pretrained-only network can put
// low policy prior on "unnatural" killer moves (e.g. a queen jump across the
// board), leaving the mate branch unvisited even at several thousand sims.
//
// depth=1 catches mate-in-1 hangs (the common case): after `candidate`, is
// there any opponent reply that delivers immediate checkmate? Only check-
// giving replies are considered — a move that's mate must give check by
// definition.
//
// The check is cheap: ~30 opponent replies × make/unmake + in_check probe
// + (rarely) legal-move gen on the mated side. Sub-millisecond per call on
// any modern CPU, dwarfed by the NN search that just finished.
bool hangs_mate(Position& pos, Move candidate, int depth = 1);

}  // namespace uci
