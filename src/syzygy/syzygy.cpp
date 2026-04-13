#include "syzygy/syzygy.h"
#include "core/bitboard.h"
#include "core/types.h"

extern "C" {
#include "tbprobe.h"
}

#include <atomic>
#include <mutex>

namespace syzygy {

namespace {
std::atomic<bool> g_ready{false};
std::atomic<uint64_t> g_hits{0};
std::mutex g_init_mutex;
} // namespace

int TableBase::init(const std::string& path) {
    std::lock_guard<std::mutex> lk(g_init_mutex);
    bool ok = tb_init(path.c_str());
    if (!ok) {
        g_ready.store(false, std::memory_order_release);
        return -1;
    }
    g_ready.store(TB_LARGEST > 0, std::memory_order_release);
    return static_cast<int>(TB_LARGEST);
}

void TableBase::shutdown() {
    std::lock_guard<std::mutex> lk(g_init_mutex);
    tb_free();
    g_ready.store(false, std::memory_order_release);
}

bool TableBase::ready() {
    return g_ready.load(std::memory_order_acquire);
}

int TableBase::max_pieces() {
    return ready() ? static_cast<int>(TB_LARGEST) : 0;
}

uint64_t TableBase::hits() {
    return g_hits.load(std::memory_order_relaxed);
}

ProbeResult TableBase::probe_wdl(const Position& pos) {
    ProbeResult r;
    if (!ready()) return r;

    Bitboard occ = pos.occupied();
    int piece_count = popcount(occ);
    if (piece_count > static_cast<int>(TB_LARGEST)) return r;

    // Fathom's tb_probe_wdl rejects positions with castling rights or a
    // nonzero halfmove clock (correct: the 50-move clock would invalidate
    // a TB cursor result mid-search). Return false here so the caller
    // skips and lets the NN/search take over.
    if (pos.castling_rights() != NO_CASTLING) return r;
    if (pos.halfmove_clock() != 0) return r;

    Bitboard white   = pos.pieces(WHITE);
    Bitboard black   = pos.pieces(BLACK);
    Bitboard kings   = pos.pieces(WHITE, KING)   | pos.pieces(BLACK, KING);
    Bitboard queens  = pos.pieces(WHITE, QUEEN)  | pos.pieces(BLACK, QUEEN);
    Bitboard rooks   = pos.pieces(WHITE, ROOK)   | pos.pieces(BLACK, ROOK);
    Bitboard bishops = pos.pieces(WHITE, BISHOP) | pos.pieces(BLACK, BISHOP);
    Bitboard knights = pos.pieces(WHITE, KNIGHT) | pos.pieces(BLACK, KNIGHT);
    Bitboard pawns   = pos.pieces(WHITE, PAWN)   | pos.pieces(BLACK, PAWN);

    unsigned ep = (pos.ep_square() == NO_SQUARE) ? 0u : static_cast<unsigned>(pos.ep_square());
    bool turn_white = (pos.side_to_move() == WHITE);

    unsigned wdl = tb_probe_wdl(
        white, black, kings, queens, rooks, bishops, knights, pawns,
        /*rule50*/ 0u,           // pre-checked above
        /*castling*/ 0u,         // pre-checked above
        ep,
        turn_white
    );

    if (wdl == TB_RESULT_FAILED) return r;

    r.hit = true;
    g_hits.fetch_add(1, std::memory_order_relaxed);

    switch (wdl) {
        case TB_WIN:           r.wdl = WDL::WIN;  break;
        case TB_LOSS:          r.wdl = WDL::LOSS; break;
        // Treat 50-move-rule edge cases as draws — safe default.
        case TB_DRAW:
        case TB_BLESSED_LOSS:
        case TB_CURSED_WIN:
        default:               r.wdl = WDL::DRAW; break;
    }
    return r;
}

} // namespace syzygy
