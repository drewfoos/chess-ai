#pragma once
#include <cstdint>
#include <optional>
#include <string>
#include <vector>

namespace uci {

// Per-mille win/draw/loss from the side-to-move's perspective.
// Emitted as "wdl <w> <d> <l>" when UCI_ShowWDL is enabled. Values sum to 1000.
struct WDL {
    int w = 0;
    int d = 0;
    int l = 0;
};

// One line of "info ..." output. All fields except depth are optional; the
// responder emits whichever are populated. Pattern mirrors Lc0's ThinkingInfo
// (engines/lc0/src/chess/uciloop.cc) — search publishes a struct, responder
// owns the text format. Keeps uci.cpp free of format strings.
struct ThinkingInfo {
    int depth = 0;
    std::optional<int> seldepth;
    int64_t nodes = 0;
    int64_t nps = 0;
    int64_t time_ms = 0;

    // At most one of score_cp / mate_plies should be set. mate_plies is signed
    // from STM's perspective (positive = STM mates, negative = STM is mated)
    // and is rendered as full moves via Stockfish's ceiling convention
    // (see stdout_responder.cpp::format_score).
    std::optional<int> score_cp;
    std::optional<int> mate_plies;

    std::optional<WDL> wdl;
    std::optional<int> hashfull;   // tree/cache occupancy, per-mille
    std::optional<int> tbhits;     // Syzygy probe hits
    std::optional<int> multipv;    // 1-based index when MultiPV > 1

    std::vector<std::string> pv;   // UCI move strings, longest-first
    std::string comment;           // emitted as a leading "info string <comment>"
};

struct BestMoveInfo {
    std::string bestmove;                  // UCI move string, or "0000" when none
    std::optional<std::string> ponder;     // Optional ponder hint (second PV move)
};

// Output sink for the UCI handler. UCI protocol lines go through one of four
// entry points; the concrete implementation decides where they actually land
// (stdout for real play, a stringstream for tests, a forwarder for ponder).
class Responder {
public:
    virtual ~Responder() = default;

    virtual void OnInfo(const ThinkingInfo& info) = 0;
    virtual void OnBestMove(const BestMoveInfo& info) = 0;

    // "info string <msg>" — free-form diagnostic channel. Most GUIs display
    // these in a log panel; some ignore them.
    virtual void OnString(const std::string& msg) = 0;

    // Raw protocol lines the UCI handshake emits verbatim: "id name ...",
    // "id author ...", "option name ... type ...", "uciok", "readyok".
    virtual void OnRaw(const std::string& line) = 0;
};

}  // namespace uci
