#pragma once
#include "core/position.h"
#include <atomic>
#include <string>

namespace syzygy {

// Result of a tablebase WDL probe, oriented from side-to-move perspective.
enum class WDL : int8_t {
    LOSS = -1,
    DRAW = 0,
    WIN  = 1,
};

struct ProbeResult {
    bool      hit = false;
    WDL       wdl = WDL::DRAW;
};

// Process-wide singleton. Fathom keeps its tables in static globals; we
// follow the same model and just expose a thin C++ wrapper.
class TableBase {
public:
    // Initialize from a Syzygy directory (e.g. "E:/dev/chess-ai/syzygy").
    // On success returns the largest piece count supported by the loaded
    // tables (e.g. 5 for the 3-4-5 set); 0 if no tables found; -1 on error.
    static int init(const std::string& path);

    // Free Fathom's tables (idempotent).
    static void shutdown();

    // True iff init() found at least one table file.
    static bool ready();

    // Largest piece count probable, or 0 when no tables loaded.
    static int max_pieces();

    // Probe WDL for `pos`. Returns hit=false if:
    //   - tables not loaded
    //   - piece count > max_pieces()
    //   - castling rights are nonzero  (TB invalid with castling)
    //   - halfmove clock != 0          (TB invalid mid-50-move-rule)
    //   - probe failed for any other reason (bad position, missing file)
    //
    // BLESSED_LOSS and CURSED_WIN are reported as DRAW (50-move-rule safe
    // default — match Lc0 behavior).
    static ProbeResult probe_wdl(const Position& pos);

    // Diagnostics: number of successful probes since process start.
    static uint64_t hits();
};

} // namespace syzygy
