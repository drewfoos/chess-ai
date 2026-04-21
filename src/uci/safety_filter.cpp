#include "uci/safety_filter.h"
#include "core/movegen.h"

namespace uci {

namespace {

// Returns true if side-to-move is in check and has no legal reply.
bool is_checkmated(const Position& pos) {
    if (!pos.in_check()) return false;
    Move buf[MAX_MOVES];
    return generate_legal_moves(pos, buf) == 0;
}

}  // namespace

bool hangs_mate(Position& pos, Move candidate, int depth) {
    if (depth < 1) return false;

    UndoInfo u_cand;
    pos.make_move(candidate, u_cand);

    Move replies[MAX_MOVES];
    int n = generate_legal_moves(pos, replies);

    bool hangs = false;
    for (int i = 0; i < n && !hangs; i++) {
        UndoInfo u_reply;
        pos.make_move(replies[i], u_reply);

        // Mate requires check — a non-check reply can't deliver mate at any depth.
        // Side-to-move after the reply is the original mover, so pos.in_check()
        // tells us whether THE ORIGINAL MOVER is now in check (i.e. the reply
        // gave check).
        if (pos.in_check()) {
            if (depth == 1) {
                hangs = is_checkmated(pos);
            } else {
                // depth >= 2: mate-in-N means opponent has a reply such that
                // every one of our responses still hangs mate in N-1. We're
                // already at "our" position here (opponent just moved); generate
                // our escape moves and recurse.
                Move escapes[MAX_MOVES];
                int nesc = generate_legal_moves(pos, escapes);
                if (nesc == 0) {
                    hangs = true;  // already mated
                } else {
                    bool all_lose = true;
                    for (int j = 0; j < nesc && all_lose; j++) {
                        if (!hangs_mate(pos, escapes[j], depth - 1)) {
                            all_lose = false;
                        }
                    }
                    hangs = all_lose;
                }
            }
        }

        pos.unmake_move(replies[i], u_reply);
    }

    pos.unmake_move(candidate, u_cand);
    return hangs;
}

}  // namespace uci
