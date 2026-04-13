#include <gtest/gtest.h>
#include "syzygy/syzygy.h"
#include "core/position.h"
#include "core/attacks.h"
#include <filesystem>

namespace {

// Find the repo's syzygy directory by walking up from CWD. Tests run from
// the build dir; the tablebase set lives at <repo>/syzygy.
std::string find_syzygy_dir() {
    namespace fs = std::filesystem;
    fs::path p = fs::current_path();
    for (int i = 0; i < 6; i++) {
        fs::path candidate = p / "syzygy";
        if (fs::exists(candidate / "KQvK.rtbw")) return candidate.string();
        if (!p.has_parent_path()) break;
        p = p.parent_path();
    }
    return {};
}

class SyzygyTest : public ::testing::Test {
protected:
    static void SetUpTestSuite() {
        attacks::init();
        std::string dir = find_syzygy_dir();
        if (!dir.empty()) {
            syzygy::TableBase::init(dir);
        }
    }
    static void TearDownTestSuite() {
        syzygy::TableBase::shutdown();
    }
};

TEST_F(SyzygyTest, InitFindsTablebases) {
    if (!syzygy::TableBase::ready()) {
        GTEST_SKIP() << "syzygy tablebases not present in repo (skipping)";
    }
    EXPECT_GE(syzygy::TableBase::max_pieces(), 5);
}

TEST_F(SyzygyTest, KQvKIsWinForWhite) {
    if (!syzygy::TableBase::ready()) GTEST_SKIP();
    Position pos;
    pos.set_fen("4k3/8/8/8/8/8/8/3QK3 w - - 0 1");
    auto r = syzygy::TableBase::probe_wdl(pos);
    ASSERT_TRUE(r.hit);
    EXPECT_EQ(r.wdl, syzygy::WDL::WIN);
}

TEST_F(SyzygyTest, KQvKIsLossForBlack) {
    if (!syzygy::TableBase::ready()) GTEST_SKIP();
    Position pos;
    pos.set_fen("4k3/8/8/8/8/8/8/3QK3 b - - 0 1");
    auto r = syzygy::TableBase::probe_wdl(pos);
    ASSERT_TRUE(r.hit);
    EXPECT_EQ(r.wdl, syzygy::WDL::LOSS);
}

TEST_F(SyzygyTest, KvKIsDraw) {
    if (!syzygy::TableBase::ready()) GTEST_SKIP();
    Position pos;
    pos.set_fen("4k3/8/8/8/8/8/8/4K3 w - - 0 1");
    auto r = syzygy::TableBase::probe_wdl(pos);
    ASSERT_TRUE(r.hit);
    EXPECT_EQ(r.wdl, syzygy::WDL::DRAW);
}

TEST_F(SyzygyTest, SkipsWhenHalfmoveClockNonzero) {
    if (!syzygy::TableBase::ready()) GTEST_SKIP();
    Position pos;
    // KQvK but with halfmove clock 5 — Fathom's contract requires 0.
    pos.set_fen("4k3/8/8/8/8/8/8/3QK3 w - - 5 1");
    auto r = syzygy::TableBase::probe_wdl(pos);
    EXPECT_FALSE(r.hit);
}

TEST_F(SyzygyTest, SkipsWhenCastlingRightsPresent) {
    if (!syzygy::TableBase::ready()) GTEST_SKIP();
    Position pos;
    // 5-piece position with castling rights — should bail out.
    pos.set_fen("4k3/8/8/8/8/8/8/R3K3 w Q - 0 1");
    auto r = syzygy::TableBase::probe_wdl(pos);
    EXPECT_FALSE(r.hit);
}

TEST_F(SyzygyTest, SkipsWhenPieceCountExceedsMax) {
    if (!syzygy::TableBase::ready()) GTEST_SKIP();
    Position pos;
    pos.set_fen("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1");
    auto r = syzygy::TableBase::probe_wdl(pos);
    EXPECT_FALSE(r.hit);
}

} // namespace
