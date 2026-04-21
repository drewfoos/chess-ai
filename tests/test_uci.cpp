#include <gtest/gtest.h>
#include "uci/time_manager.h"

TEST(TimeManager, MovetimeConvertsToIterations) {
    uci::TimeControl tc;
    tc.movetime_ms = 1000;
    auto alloc = uci::allocate_time(tc, WHITE, 500);
    EXPECT_EQ(alloc.iterations, 500);
    EXPECT_EQ(alloc.soft_time_ms, 1000);
}

TEST(TimeManager, InfiniteReturnsLargeCount) {
    uci::TimeControl tc;
    tc.infinite = true;
    auto alloc = uci::allocate_time(tc, WHITE);
    EXPECT_GE(alloc.iterations, 100000);
}

TEST(TimeManager, MaxNodesPassedThrough) {
    uci::TimeControl tc;
    tc.max_nodes = 400;
    auto alloc = uci::allocate_time(tc, WHITE);
    EXPECT_EQ(alloc.iterations, 400);
}

TEST(TimeManager, ClockBasedAllocation) {
    uci::TimeControl tc;
    tc.wtime_ms = 60000;
    tc.winc_ms = 1000;
    auto alloc = uci::allocate_time(tc, WHITE, 500);
    EXPECT_GT(alloc.iterations, 500);
    EXPECT_LT(alloc.iterations, 3000);
}

TEST(TimeManager, NeverExceedsHalfRemainingTime) {
    uci::TimeControl tc;
    tc.wtime_ms = 1000;
    tc.winc_ms = 0;
    auto alloc = uci::allocate_time(tc, WHITE, 500);
    EXPECT_LE(alloc.soft_time_ms, 500);
}

TEST(TimeManager, BlackUsesBlackClock) {
    uci::TimeControl tc;
    tc.wtime_ms = 60000;
    tc.btime_ms = 10000;
    auto alloc = uci::allocate_time(tc, BLACK, 500);
    EXPECT_LT(alloc.iterations, 500);
}

TEST(TimeManager, HardCapsAtQuarterOfClock) {
    // Pathological: tiny base time with massive increment. Without a hard
    // upper bound, increment-aware allocation would burn the entire base
    // clock on one move, leaving us flagged before the inc ticks back in.
    // Cap protects against this regardless of how favorable the math looks.
    uci::TimeControl tc;
    tc.wtime_ms = 1000;
    tc.winc_ms = 100000;
    auto alloc = uci::allocate_time(tc, WHITE, 500);
    EXPECT_LE(alloc.soft_time_ms, 250);
}

TEST(TimeManager, SuddenDeathConservativeWithoutIncrement) {
    // Sudden death (no increment) means each move's budget permanently
    // shrinks the clock. We need to assume more remaining moves to keep
    // the per-move budget small. Compare two identical wtimes — the one
    // with NO increment should get a strictly smaller budget than the
    // one WITH increment.
    uci::TimeControl tc_no_inc, tc_with_inc;
    tc_no_inc.wtime_ms = tc_with_inc.wtime_ms = 60000;
    tc_no_inc.winc_ms = 0;
    tc_with_inc.winc_ms = 1000;
    auto a_no_inc = uci::allocate_time(tc_no_inc, WHITE, 500);
    auto a_with_inc = uci::allocate_time(tc_with_inc, WHITE, 500);
    EXPECT_LT(a_no_inc.soft_time_ms, a_with_inc.soft_time_ms);
}

TEST(TimeManager, LeavesEmergencyReserveInTimeScramble) {
    // In a low-clock time scramble (sub-second), the time manager must
    // leave a margin on the clock — not propose a budget that reaches
    // straight to the buzzer. Otherwise GUI latency or a single slow NN
    // batch flags us. With wtime=200ms, propose well under 100ms.
    uci::TimeControl tc;
    tc.wtime_ms = 200;
    auto alloc = uci::allocate_time(tc, WHITE, 500);
    EXPECT_LT(alloc.soft_time_ms, 30);
}

// --- UCI Protocol Handler Tests ---

#include "uci/uci.h"
#include "core/attacks.h"

class UCITest : public ::testing::Test {
protected:
    static void SetUpTestSuite() { attacks::init(); }
};

TEST_F(UCITest, UciCommandReturnsUciok) {
    std::istringstream in("uci\nquit\n");
    std::ostringstream out;
    mcts::RandomEvaluator eval;
    uci::UCIHandler handler(eval, {}, in, out);
    handler.loop();
    EXPECT_NE(out.str().find("id name ChessAI"), std::string::npos);
    EXPECT_NE(out.str().find("uciok"), std::string::npos);
    EXPECT_NE(out.str().find("option name SyzygyPath"), std::string::npos);
}

TEST_F(UCITest, SetOptionSyzygyPathInvalidEmitsInfo) {
    // Pointing at a non-tablebase directory should not crash; engine
    // emits an "info string" telling the GUI nothing was loaded.
    std::istringstream in("setoption name SyzygyPath value /this/path/does/not/exist\nquit\n");
    std::ostringstream out;
    mcts::RandomEvaluator eval;
    uci::UCIHandler handler(eval, {}, in, out);
    handler.loop();
    EXPECT_NE(out.str().find("info string Syzygy"), std::string::npos);
}

TEST_F(UCITest, IsReadyReturnsReadyok) {
    std::istringstream in("isready\nquit\n");
    std::ostringstream out;
    mcts::RandomEvaluator eval;
    uci::UCIHandler handler(eval, {}, in, out);
    handler.loop();
    EXPECT_NE(out.str().find("readyok"), std::string::npos);
}

TEST_F(UCITest, PositionStartposAndGo) {
    std::istringstream in("position startpos\ngo nodes 50\nquit\n");
    std::ostringstream out;
    mcts::RandomEvaluator eval;
    mcts::SearchParams params;
    params.add_noise = false;
    uci::UCIHandler handler(eval, params, in, out);
    handler.loop();
    EXPECT_NE(out.str().find("bestmove"), std::string::npos);
}

TEST_F(UCITest, PositionStartposMoves) {
    std::istringstream in("position startpos moves e2e4 e7e5\ngo nodes 50\nquit\n");
    std::ostringstream out;
    mcts::RandomEvaluator eval;
    mcts::SearchParams params;
    params.add_noise = false;
    uci::UCIHandler handler(eval, params, in, out);
    handler.loop();
    EXPECT_NE(out.str().find("bestmove"), std::string::npos);
}

TEST_F(UCITest, PositionFen) {
    std::istringstream in(
        "position fen rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1\n"
        "go nodes 50\nquit\n");
    std::ostringstream out;
    mcts::RandomEvaluator eval;
    mcts::SearchParams params;
    params.add_noise = false;
    uci::UCIHandler handler(eval, params, in, out);
    handler.loop();
    EXPECT_NE(out.str().find("bestmove"), std::string::npos);
}

TEST_F(UCITest, GoInfiniteAndStop) {
    std::istringstream in("position startpos\ngo infinite\nstop\nquit\n");
    std::ostringstream out;
    mcts::RandomEvaluator eval;
    mcts::SearchParams params;
    params.add_noise = false;
    uci::UCIHandler handler(eval, params, in, out);
    handler.loop();
    EXPECT_NE(out.str().find("bestmove"), std::string::npos);
}

TEST_F(UCITest, UciAdvertisesGuiFriendlyOptions) {
    // Arena/Lichess routinely try to set Hash/Threads/Ponder/Move Overhead.
    // Missing advertisements make some GUIs refuse the engine entirely.
    std::istringstream in("uci\nquit\n");
    std::ostringstream out;
    mcts::RandomEvaluator eval;
    uci::UCIHandler handler(eval, {}, in, out);
    handler.loop();
    auto s = out.str();
    EXPECT_NE(s.find("option name Hash"), std::string::npos);
    EXPECT_NE(s.find("option name Threads"), std::string::npos);
    EXPECT_NE(s.find("option name UCI_Chess960"), std::string::npos);
    EXPECT_NE(s.find("option name Ponder"), std::string::npos);
    EXPECT_NE(s.find("option name Move Overhead"), std::string::npos);
}

TEST_F(UCITest, SetOptionAcceptsMultiWordNames) {
    // "Move Overhead" has a space — the parser must handle that. Also verify
    // it doesn't blow up on Hash/Threads/Ponder (which we accept but ignore).
    std::istringstream in(
        "setoption name Hash value 512\n"
        "setoption name Threads value 4\n"
        "setoption name Ponder value false\n"
        "setoption name Move Overhead value 200\n"
        "setoption name Iterations value 123\n"
        "position startpos\n"
        "go nodes 10\n"
        "quit\n");
    std::ostringstream out;
    mcts::RandomEvaluator eval;
    uci::UCIHandler handler(eval, {}, in, out);
    handler.loop();
    // Nothing crashed and we still produced a move.
    EXPECT_NE(out.str().find("bestmove"), std::string::npos);
}

TEST_F(UCITest, InfoLineIncludesDepthAndTime) {
    // Arena/Lichess display Depth and Time columns — make sure we emit them.
    std::istringstream in(
        "position startpos\ngo nodes 200\nquit\n");
    std::ostringstream out;
    mcts::RandomEvaluator eval;
    mcts::SearchParams params;
    params.add_noise = false;
    uci::UCIHandler handler(eval, params, in, out);
    handler.loop();
    auto s = out.str();
    EXPECT_NE(s.find("info"), std::string::npos);
    EXPECT_NE(s.find("depth "), std::string::npos);
    EXPECT_NE(s.find("time "), std::string::npos);
}

TEST_F(UCITest, FinalInfoLineReportsSeldepth) {
    // seldepth is the deepest descent of the search; it must be >0 after any
    // run that completes at least one gather. Without it GUIs like CuteChess
    // hide the selective-depth column.
    std::istringstream in("position startpos\ngo nodes 200\n");
    std::ostringstream out;
    mcts::RandomEvaluator eval;
    mcts::SearchParams params;
    params.add_noise = false;
    uci::UCIHandler handler(eval, params, in, out);
    handler.loop();
    handler.wait_for_search();
    EXPECT_NE(out.str().find("seldepth "), std::string::npos);
}

TEST_F(UCITest, PvWalksBeyondBestMove) {
    // With a 200-node random search the tree goes several plies deep along
    // the most-visited branch. The PV must report that walk, not just echo
    // the bestmove — GUIs use PV length as a quality signal and ponder
    // pairing uses PV[1].
    std::istringstream in("position startpos\ngo nodes 200\n");
    std::ostringstream out;
    mcts::RandomEvaluator eval;
    mcts::SearchParams params;
    params.add_noise = false;
    uci::UCIHandler handler(eval, params, in, out);
    handler.loop();
    handler.wait_for_search();
    auto s = out.str();
    // Find the final "info" line (the one just before bestmove).
    auto bm_pos = s.find("bestmove ");
    ASSERT_NE(bm_pos, std::string::npos);
    auto info_pos = s.rfind("info", bm_pos);
    ASSERT_NE(info_pos, std::string::npos);
    std::string final_info = s.substr(info_pos, bm_pos - info_pos);
    auto pv_pos = final_info.find(" pv ");
    ASSERT_NE(pv_pos, std::string::npos) << "final info: " << final_info;
    std::string pv_tail = final_info.substr(pv_pos + 4);
    int move_count = 0;
    size_t i = 0;
    while (i < pv_tail.size()) {
        while (i < pv_tail.size() && pv_tail[i] == ' ') i++;
        if (i >= pv_tail.size() || pv_tail[i] == '\n') break;
        move_count++;
        while (i < pv_tail.size() && pv_tail[i] != ' ' && pv_tail[i] != '\n') i++;
    }
    EXPECT_GT(move_count, 1) << "PV was: " << pv_tail;
}

TEST_F(UCITest, IllegalMoveDoesNotCrashEngine) {
    // parse_uci_move throws std::runtime_error on unparseable / illegal moves.
    // Without a handler in loop() the exception propagates out and terminates
    // the process. A buggy GUI or unexpected promotion format (e.g. capital
    // 'Q') would kill the engine mid-game. Engine must stay alive, emit an
    // "info string" telling the GUI we couldn't apply the move, and still
    // respond to subsequent isready/go commands.
    std::istringstream in(
        "uci\n"
        "position startpos moves xxxx\n"
        "isready\n"
        "position startpos\n"
        "go nodes 50\n"
        "quit\n");
    std::ostringstream out;
    mcts::RandomEvaluator eval;
    mcts::SearchParams params;
    params.add_noise = false;
    uci::UCIHandler handler(eval, params, in, out);
    handler.loop();
    auto s = out.str();
    // Engine must still be alive after the bad command.
    EXPECT_NE(s.find("readyok"), std::string::npos) << s;
    EXPECT_NE(s.find("bestmove"), std::string::npos) << s;
    // Diagnostic telling the GUI we rejected the move.
    EXPECT_NE(s.find("info string"), std::string::npos) << s;
}

TEST_F(UCITest, IllegalMoveMidSequenceRevertsState) {
    // `position startpos moves e2e4 xxxx` must be atomic: if any move in the
    // sequence is illegal, the whole command is rejected and current_pos_ is
    // left unchanged. Otherwise e2e4 gets silently applied, and a subsequent
    // `go` with no fresh `position` reset would search from black-to-move
    // while the GUI thinks it's white-to-move — move desync across the wire.
    //
    // We verify by running `go` AFTER the failing command (no intervening
    // `position`): bestmove must come from rank 2 (a white opening move),
    // not rank 7 (which would prove e2e4 was applied and it's now black's turn).
    std::istringstream in(
        "uci\n"
        "position startpos moves e2e4 xxxx\n"
        "go nodes 50\n"
        "quit\n");
    std::ostringstream out;
    mcts::RandomEvaluator eval;
    mcts::SearchParams params;
    params.add_noise = false;
    uci::UCIHandler handler(eval, params, in, out);
    handler.loop();
    auto s = out.str();
    auto bm = s.find("bestmove ");
    ASSERT_NE(bm, std::string::npos) << s;
    std::string move = s.substr(bm + 9, 4);
    // From-rank must be '2' (white pawn/piece move from rank 2) for white-to-
    // move openings, OR other rank-1 piece moves (b1, g1, etc). Key constraint:
    // from-rank is '1' or '2', NOT '7' or '8'.
    EXPECT_TRUE(move[1] == '1' || move[1] == '2')
        << "expected white move (from rank 1 or 2), got: " << move << "\nfull output:\n" << s;
}

TEST_F(UCITest, SetOptionIterationsClampsNegative) {
    // Iterations is advertised as min 1 max 100000. Sending -5 must not cause
    // the next search to spin forever, divide by zero, or produce bestmove 0000.
    std::istringstream in(
        "setoption name Iterations value -5\n"
        "position startpos\n"
        "go\n"
        "quit\n");
    std::ostringstream out;
    mcts::RandomEvaluator eval;
    uci::UCIHandler handler(eval, {}, in, out);
    handler.loop();
    auto s = out.str();
    auto bm = s.find("bestmove ");
    ASSERT_NE(bm, std::string::npos) << s;
    std::string move = s.substr(bm + 9, 4);
    EXPECT_NE(move, "0000") << "negative Iterations caused empty bestmove: " << s;
}

TEST_F(UCITest, UciShowWdlAdvertisedAsOption) {
    // GUI-side UCI_ShowWDL toggle is what Lichess / Arena look for to request
    // per-mille WDL in info lines. Missing advertisement means they never
    // enable it.
    std::istringstream in("uci\nquit\n");
    std::ostringstream out;
    mcts::RandomEvaluator eval;
    uci::UCIHandler handler(eval, {}, in, out);
    handler.loop();
    EXPECT_NE(out.str().find("option name UCI_ShowWDL"), std::string::npos);
}

TEST_F(UCITest, UciShowWdlOffByDefault) {
    // Default: no wdl field in info line. Some GUIs treat its presence as a
    // signal to swap their display column, so don't emit it unprompted.
    std::istringstream in("position startpos\ngo nodes 50\n");
    std::ostringstream out;
    mcts::RandomEvaluator eval;
    mcts::SearchParams params;
    params.add_noise = false;
    uci::UCIHandler handler(eval, params, in, out);
    handler.loop();
    handler.wait_for_search();
    EXPECT_EQ(out.str().find(" wdl "), std::string::npos) << out.str();
}

TEST_F(UCITest, UciShowWdlEmitsPerMille) {
    // With UCI_ShowWDL on, the final info line should carry a `wdl W D L`
    // triplet summing to ~1000 (integer rounding can lose 1-2 per-mille).
    std::istringstream in(
        "setoption name UCI_ShowWDL value true\n"
        "position startpos\n"
        "go nodes 50\n");
    std::ostringstream out;
    mcts::RandomEvaluator eval;
    mcts::SearchParams params;
    params.add_noise = false;
    uci::UCIHandler handler(eval, params, in, out);
    handler.loop();
    handler.wait_for_search();
    auto s = out.str();
    auto pos = s.find(" wdl ");
    ASSERT_NE(pos, std::string::npos) << s;
    int w = 0, d = 0, l = 0;
    int scanned = std::sscanf(s.c_str() + pos, " wdl %d %d %d", &w, &d, &l);
    ASSERT_EQ(scanned, 3) << s;
    EXPECT_GE(w, 0); EXPECT_LE(w, 1000);
    EXPECT_GE(d, 0); EXPECT_LE(d, 1000);
    EXPECT_GE(l, 0); EXPECT_LE(l, 1000);
    int sum = w + d + l;
    EXPECT_GE(sum, 998);
    EXPECT_LE(sum, 1000);
}

TEST_F(UCITest, MultiPvAdvertisedAsOption) {
    std::istringstream in("uci\nquit\n");
    std::ostringstream out;
    mcts::RandomEvaluator eval;
    uci::UCIHandler handler(eval, {}, in, out);
    handler.loop();
    EXPECT_NE(out.str().find("option name MultiPV"), std::string::npos);
}

TEST_F(UCITest, MultiPvEmitsTopNLines) {
    // With MultiPV=3, the final round of info output must contain three lines
    // with `multipv 1`, `multipv 2`, `multipv 3`. GUIs expect all three in the
    // same "refresh" before bestmove (Lichess analysis, Scid vs PC).
    std::istringstream in(
        "setoption name MultiPV value 3\n"
        "position startpos\n"
        "go nodes 200\n");
    std::ostringstream out;
    mcts::RandomEvaluator eval;
    mcts::SearchParams params;
    params.add_noise = false;
    uci::UCIHandler handler(eval, params, in, out);
    handler.loop();
    handler.wait_for_search();
    auto s = out.str();
    EXPECT_NE(s.find("multipv 1"), std::string::npos) << s;
    EXPECT_NE(s.find("multipv 2"), std::string::npos) << s;
    EXPECT_NE(s.find("multipv 3"), std::string::npos) << s;
    EXPECT_NE(s.find("bestmove"), std::string::npos) << s;
}

TEST_F(UCITest, MultiPvDefaultEmitsNoMultipvField) {
    // With MultiPV unset (default=1), info lines must NOT include `multipv`
    // at all — some GUIs switch display mode when they see it.
    std::istringstream in("position startpos\ngo nodes 200\n");
    std::ostringstream out;
    mcts::RandomEvaluator eval;
    mcts::SearchParams params;
    params.add_noise = false;
    uci::UCIHandler handler(eval, params, in, out);
    handler.loop();
    handler.wait_for_search();
    EXPECT_EQ(out.str().find("multipv "), std::string::npos);
}

TEST_F(UCITest, GoPonderDoesNotEmitBestMoveUntilStop) {
    // During `go ponder`, UCI spec says the engine MUST NOT emit bestmove
    // until either ponderhit or stop arrives. Otherwise the GUI sees a
    // spurious early bestmove when it was expecting the engine to keep
    // searching speculatively.
    std::istringstream in(
        "position startpos\n"
        "go ponder wtime 60000 btime 60000\n"
        "stop\n"
        "quit\n");
    std::ostringstream out;
    mcts::RandomEvaluator eval;
    mcts::SearchParams params;
    params.add_noise = false;
    uci::UCIHandler handler(eval, params, in, out);
    handler.loop();
    auto s = out.str();
    // One bestmove total, emitted AFTER the stop arrives (which handle_stop
    // joins, so it's necessarily emitted before loop() returns).
    EXPECT_NE(s.find("bestmove"), std::string::npos) << s;
    // Also: the search must not have self-terminated before stop — in ponder
    // mode there's no deadline, so a finite node count can't trigger emit.
    // We verify by counting bestmove occurrences (should be exactly 1).
    auto first = s.find("bestmove");
    auto second = s.find("bestmove", first + 1);
    EXPECT_EQ(second, std::string::npos) << "multiple bestmoves: " << s;
}

TEST_F(UCITest, PonderHitConvertsToTimedSearch) {
    // ponderhit transitions the in-flight ponder search into a normal timed
    // search using the original go's wtime/btime. Bestmove emission must
    // happen within a few hundred ms even though no `stop` is ever sent.
    std::istringstream in(
        "position startpos\n"
        "go ponder wtime 200 btime 200\n"
        "ponderhit\n");
    std::ostringstream out;
    mcts::RandomEvaluator eval;
    mcts::SearchParams params;
    params.add_noise = false;
    uci::UCIHandler handler(eval, params, in, out);
    handler.loop();
    handler.wait_for_search();
    auto s = out.str();
    EXPECT_NE(s.find("bestmove"), std::string::npos) << s;
}

TEST_F(UCITest, PonderHitOutsideSearchIsHarmless) {
    // Spurious ponderhit (no search running) must not crash or hang.
    std::istringstream in("ponderhit\nisready\nquit\n");
    std::ostringstream out;
    mcts::RandomEvaluator eval;
    uci::UCIHandler handler(eval, {}, in, out);
    handler.loop();
    EXPECT_NE(out.str().find("readyok"), std::string::npos);
}

TEST_F(UCITest, MateInOneEmitsScoreMate) {
    // Back-rank mate: white plays Ra8#. MCTS-solver marks the child as
    // terminal (opponent has no legal reply), propagates up to the root,
    // and the final info line should report `score mate N` instead of cp.
    std::istringstream in(
        // Back-rank mate-in-1: white rook a1 has a clear a-file, black king on
        // g8 is smothered by its own pawn shield (f7/g7/h7). Ra8 is mate.
        "position fen 6k1/5ppp/8/8/8/8/1PP5/R5K1 w - - 0 1\n"
        "go nodes 5000\n");
    std::ostringstream out;
    mcts::RandomEvaluator eval;
    mcts::SearchParams params;
    params.add_noise = false;
    uci::UCIHandler handler(eval, params, in, out);
    handler.loop();
    handler.wait_for_search();
    auto s = out.str();
    EXPECT_NE(s.find("score mate"), std::string::npos) << s;
    // Bestmove should be a1a8 (the mating rook move).
    EXPECT_NE(s.find("bestmove a1a8"), std::string::npos) << s;
}

// --- Tactical Safety Filter Tests ---
//
// Catches the mate-in-1 blunder class demonstrated in Lichess game HAIonk2l:
// a pretrained-only network placed ~0 policy prior on Black's Qh1# reply, so
// with ~5k sims after White played 24.g3?? the mate branch was never expanded.
// The filter re-checks the top MCTS candidate against all opponent replies
// that give check, rejecting it if any delivers mate.

#include "uci/safety_filter.h"
#include "core/move_parser.h"

class SafetyFilterTest : public ::testing::Test {
protected:
    static void SetUpTestSuite() { attacks::init(); }
};

namespace {

Move uci_to_move(Position& pos, const std::string& s) {
    try {
        return parse_uci_move(pos, s);
    } catch (const std::exception& e) {
        ADD_FAILURE() << "failed to parse UCI move " << s << ": " << e.what();
        return Move::none();
    }
}

}  // namespace

TEST_F(SafetyFilterTest, StartPosE4DoesNotHangMate) {
    // Sanity: a normal opening move has no mate-in-1 reply. If this fails
    // the filter is flagging safe moves and would make the engine unplayable.
    Position pos;
    pos.set_fen("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1");
    Move e4 = uci_to_move(pos, "e2e4");
    EXPECT_FALSE(uci::hangs_mate(pos, e4, 1));
}

TEST_F(SafetyFilterTest, CatchesGameBlunderG3) {
    // The actual blunder from lichess.org/HAIonk2l move 24: after 24.g3??
    // Black plays 24...Qh1# — the queen slides down the a8-h1 diagonal
    // through the square g2 just vacated by the pawn push. The fix we care
    // about: the filter must flag g3 as hanging mate.
    Position pos;
    pos.set_fen("6k1/p4p1p/5Qp1/3b4/4q3/4N3/PP3PPP/6K1 w - - 0 24");
    Move g3 = uci_to_move(pos, "g2g3");
    EXPECT_TRUE(uci::hangs_mate(pos, g3, 1));
}

TEST_F(SafetyFilterTest, SafePawnMoveH3Passes) {
    // Same blunder position — h2h3 keeps g2 defended and doesn't clear any
    // mating diagonal, so the filter should let it through. Guards against
    // the filter over-rejecting every move in a tough spot.
    Position pos;
    pos.set_fen("6k1/p4p1p/5Qp1/3b4/4q3/4N3/PP3PPP/6K1 w - - 0 24");
    Move h3 = uci_to_move(pos, "h2h3");
    EXPECT_FALSE(uci::hangs_mate(pos, h3, 1));
}

TEST_F(SafetyFilterTest, NoMateNoFalsePositive) {
    // Quiet middlegame (after 1.e4 e5 2.Nf3 Nc6): Nc3 is a standard
    // developing move with no mate threat anywhere. Belt-and-braces check
    // against the filter becoming over-eager.
    Position pos;
    pos.set_fen("r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3");
    Move nc3 = uci_to_move(pos, "b1c3");
    EXPECT_FALSE(uci::hangs_mate(pos, nc3, 1));
}
