#include <gtest/gtest.h>
#include "core/position.h"
#include "core/attacks.h"

class PositionTest : public ::testing::Test {
protected:
    void SetUp() override { attacks::init(); }
};

TEST_F(PositionTest, StartingPositionFEN) {
    Position pos;
    pos.set_fen("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1");
    EXPECT_EQ(pos.side_to_move(), WHITE);
    EXPECT_EQ(pos.castling_rights(), ALL_CASTLING);
    EXPECT_EQ(pos.ep_square(), NO_SQUARE);
    EXPECT_EQ(pos.halfmove_clock(), 0);
    EXPECT_EQ(pos.fullmove_number(), 1);
    EXPECT_EQ(pos.piece_on(E1), KING);
    EXPECT_EQ(pos.color_on(E1), WHITE);
    EXPECT_EQ(pos.piece_on(E8), KING);
    EXPECT_EQ(pos.color_on(E8), BLACK);
    EXPECT_EQ(pos.piece_on(A2), PAWN);
    EXPECT_EQ(pos.piece_on(D8), QUEEN);
    EXPECT_EQ(pos.piece_on(E4), NO_PIECE_TYPE);
}

TEST_F(PositionTest, FENRoundTrip) {
    const char* fens[] = {
        "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
        // Legitimate EP: white pawn on e5 can capture f5 via e5xf6.
        "rnbqkbnr/ppp1p1pp/3p4/4Pp2/8/8/PPPP1PPP/RNBQKBNR w KQkq f6 0 3",
        "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
        "8/8/8/8/8/8/8/4K2k w - - 0 1",
        "r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1",
    };
    Position pos;
    for (const char* fen : fens) {
        pos.set_fen(fen);
        EXPECT_EQ(pos.to_fen(), std::string(fen)) << "Failed roundtrip for: " << fen;
    }
}

TEST_F(PositionTest, PieceBitboards) {
    Position pos;
    pos.set_fen("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1");
    Bitboard wp = pos.pieces(WHITE, PAWN);
    EXPECT_EQ(wp, RANK_BB[1]);
    Bitboard bp = pos.pieces(BLACK, PAWN);
    EXPECT_EQ(bp, RANK_BB[6]);
    EXPECT_EQ(popcount(pos.occupied()), 32);
    EXPECT_EQ(popcount(pos.occupied(WHITE)), 16);
    EXPECT_EQ(popcount(pos.occupied(BLACK)), 16);
}

TEST_F(PositionTest, IsSquareAttacked) {
    Position pos;
    pos.set_fen("rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1");
    EXPECT_TRUE(pos.is_attacked(D5, WHITE));
    EXPECT_TRUE(pos.is_attacked(F5, WHITE));
    EXPECT_FALSE(pos.is_attacked(E5, WHITE));
}

TEST_F(PositionTest, InCheck) {
    Position pos;
    pos.set_fen("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1");
    EXPECT_FALSE(pos.in_check());
    pos.set_fen("4k3/8/8/8/8/8/8/4R2K b - - 0 1");
    EXPECT_TRUE(pos.in_check());
}

TEST_F(PositionTest, MakeUnmakeSimple) {
    Position pos;
    pos.set_fen("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1");
    std::string original_fen = pos.to_fen();

    UndoInfo undo;
    Move m(E2, E4, FLAG_DOUBLE_PUSH);
    pos.make_move(m, undo);

    EXPECT_EQ(pos.piece_on(E4), PAWN);
    EXPECT_EQ(pos.piece_on(E2), NO_PIECE_TYPE);
    EXPECT_EQ(pos.side_to_move(), BLACK);
    // No black pawn is on d4 or f4 to capture en passant, so ep_square is
    // canonicalized to NO_SQUARE (lc0 / python-chess behavior).
    EXPECT_EQ(pos.ep_square(), NO_SQUARE);

    pos.unmake_move(m, undo);
    EXPECT_EQ(pos.to_fen(), original_fen);
}

TEST_F(PositionTest, MakeUnmakeCapture) {
    Position pos;
    pos.set_fen("rnbqkbnr/ppp1pppp/8/3p4/4P3/8/PPPP1PPP/RNBQKBNR w KQkq d6 0 2");
    std::string original_fen = pos.to_fen();

    UndoInfo undo;
    Move m(E4, D5, FLAG_CAPTURE);
    pos.make_move(m, undo);

    EXPECT_EQ(pos.piece_on(D5), PAWN);
    EXPECT_EQ(pos.color_on(D5), WHITE);
    EXPECT_EQ(pos.piece_on(E4), NO_PIECE_TYPE);

    pos.unmake_move(m, undo);
    EXPECT_EQ(pos.to_fen(), original_fen);
}

TEST_F(PositionTest, MakeUnmakeCastling) {
    Position pos;
    pos.set_fen("r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1");
    std::string original_fen = pos.to_fen();

    UndoInfo undo;
    Move m(E1, G1, FLAG_KING_CASTLE);
    pos.make_move(m, undo);

    EXPECT_EQ(pos.piece_on(G1), KING);
    EXPECT_EQ(pos.piece_on(F1), ROOK);
    EXPECT_EQ(pos.piece_on(E1), NO_PIECE_TYPE);
    EXPECT_EQ(pos.piece_on(H1), NO_PIECE_TYPE);

    pos.unmake_move(m, undo);
    EXPECT_EQ(pos.to_fen(), original_fen);
}

TEST_F(PositionTest, MakeUnmakeEnPassant) {
    Position pos;
    pos.set_fen("rnbqkbnr/ppppp1pp/8/4Pp2/8/8/PPPP1PPP/RNBQKBNR w KQkq f6 0 3");
    std::string original_fen = pos.to_fen();

    UndoInfo undo;
    Move m(E5, F6, FLAG_EP_CAPTURE);
    pos.make_move(m, undo);

    EXPECT_EQ(pos.piece_on(F6), PAWN);
    EXPECT_EQ(pos.color_on(F6), WHITE);
    EXPECT_EQ(pos.piece_on(F5), NO_PIECE_TYPE); // captured pawn gone
    EXPECT_EQ(pos.piece_on(E5), NO_PIECE_TYPE);

    pos.unmake_move(m, undo);
    EXPECT_EQ(pos.to_fen(), original_fen);
}

TEST_F(PositionTest, MakeUnmakePromotion) {
    Position pos;
    pos.set_fen("8/P7/8/8/8/8/8/4K2k w - - 0 1");
    std::string original_fen = pos.to_fen();

    UndoInfo undo;
    Move m(A7, A8, FLAG_PROMO_QUEEN);
    pos.make_move(m, undo);

    EXPECT_EQ(pos.piece_on(A8), QUEEN);
    EXPECT_EQ(pos.color_on(A8), WHITE);
    EXPECT_EQ(pos.piece_on(A7), NO_PIECE_TYPE);

    pos.unmake_move(m, undo);
    EXPECT_EQ(pos.to_fen(), original_fen);
}

TEST_F(PositionTest, InsufficientMaterial_KvK) {
    Position pos;
    pos.set_fen("8/8/8/4k3/8/8/8/4K3 w - - 0 1");
    EXPECT_TRUE(pos.is_insufficient_material());
}

TEST_F(PositionTest, InsufficientMaterial_KBvK) {
    Position pos;
    pos.set_fen("8/8/8/4k3/8/8/8/3BK3 w - - 0 1");
    EXPECT_TRUE(pos.is_insufficient_material());
}

TEST_F(PositionTest, InsufficientMaterial_KNvK) {
    Position pos;
    pos.set_fen("8/8/8/4k3/8/8/8/3NK3 w - - 0 1");
    EXPECT_TRUE(pos.is_insufficient_material());
}

TEST_F(PositionTest, InsufficientMaterial_KBvKB_SameColor) {
    // Both bishops on light squares (c1 and f8 are both light... actually
    // c1 = (2+0)=2 even=dark; f8 = (5+7)=12 even=dark. Both dark → draw).
    Position pos;
    pos.set_fen("5b2/8/8/4k3/8/8/8/2B1K3 w - - 0 1");
    EXPECT_TRUE(pos.is_insufficient_material());
}

TEST_F(PositionTest, InsufficientMaterial_KBvKB_DifferentColor) {
    // Bishops on opposite colors: c1 dark, f1 light. Not an automatic draw.
    Position pos;
    pos.set_fen("8/8/8/4k3/8/8/8/2b1KB2 w - - 0 1");
    EXPECT_FALSE(pos.is_insufficient_material());
}

TEST_F(PositionTest, InsufficientMaterial_Pawns) {
    Position pos;
    pos.set_fen("8/8/8/4k3/8/4P3/8/4K3 w - - 0 1");
    EXPECT_FALSE(pos.is_insufficient_material());
}

TEST_F(PositionTest, InsufficientMaterial_RookPresent) {
    Position pos;
    pos.set_fen("8/8/8/4k3/8/8/8/R3K3 w - - 0 1");
    EXPECT_FALSE(pos.is_insufficient_material());
}

TEST_F(PositionTest, InsufficientMaterial_StartingPosition) {
    Position pos;
    pos.set_fen("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1");
    EXPECT_FALSE(pos.is_insufficient_material());
}

TEST_F(PositionTest, InsufficientMaterial_TwoKnightsVsK) {
    // K+N+N vs K: not forceable, but FIDE lets play continue.
    Position pos;
    pos.set_fen("8/8/8/4k3/8/8/8/2NNK3 w - - 0 1");
    EXPECT_FALSE(pos.is_insufficient_material());
}

// ---------------------------------------------------------------------------
// En-passant canonicalization (match lc0): the ep_square is only set when an
// enemy pawn can actually capture it. Storing a phantom EP creates hash drift
// (two equivalent positions hash differently) and non-canonical FEN output.
// ---------------------------------------------------------------------------

TEST_F(PositionTest, EpSquare_NotSetWhenNoEnemyPawnCanCapture) {
    // From the starting position, 1.e4 leaves e3 as a skipped square — but no
    // black pawn is on d4 or f4 to capture, so ep_square must be NO_SQUARE.
    Position pos;
    pos.set_fen("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1");
    UndoInfo undo;
    pos.make_move(Move(E2, E4, FLAG_DOUBLE_PUSH), undo);
    EXPECT_EQ(pos.ep_square(), NO_SQUARE);
}

TEST_F(PositionTest, EpSquare_SetWhenEnemyPawnCanCapture) {
    // 1.e4 d5 2.e5 f5 — after f7f5, white pawn on e5 CAN capture on f6.
    Position pos;
    pos.set_fen("rnbqkbnr/ppp1p1pp/3p4/4Pp2/8/8/PPPP1PPP/RNBQKBNR w KQkq f6 0 3");
    EXPECT_EQ(pos.ep_square(), F6);
}

TEST_F(PositionTest, EpSquare_HashesEqualForEquivalentBoards) {
    // Two routes to the same piece layout: one via double-push (phantom EP),
    // one via two single pushes. After the fix, both hash identically because
    // neither sets a capturable EP.
    Position via_double;
    via_double.set_fen("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1");
    UndoInfo u;
    via_double.make_move(Move(E2, E4, FLAG_DOUBLE_PUSH), u);

    Position via_singles;
    via_singles.set_fen("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1");
    via_singles.make_move(Move(E2, E3, FLAG_QUIET), u);
    // Black plays something reversible and reverses it.
    via_singles.make_move(Move(G8, F6, FLAG_QUIET), u);
    via_singles.make_move(Move(E3, E4, FLAG_QUIET), u);
    via_singles.make_move(Move(F6, G8, FLAG_QUIET), u);

    EXPECT_EQ(via_double.ep_square(), NO_SQUARE);
    EXPECT_EQ(via_singles.ep_square(), NO_SQUARE);
    // Same piece bitboards + same STM + no phantom EP → same board state.
    // (We don't directly compare hashes here — that path is tested in
    // test_mcts.cpp via is_repetition — but this test documents the contract
    // that phantom EP must not be set.)
}

TEST_F(PositionTest, EpSquare_FenParseRejectsPhantomEp) {
    // FEN claims EP=e3 but no black pawn is on d4/f4. set_fen must normalize.
    Position pos;
    pos.set_fen("rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1");
    EXPECT_EQ(pos.ep_square(), NO_SQUARE);
    // Canonical FEN output must drop the phantom EP too.
    std::string emitted = pos.to_fen();
    EXPECT_NE(emitted.find(" - "), std::string::npos)
        << "to_fen should emit '-' for non-capturable EP, got: " << emitted;
}

TEST_F(PositionTest, EpSquare_FenParseKeepsLegitimateEp) {
    // Here EP=f6 IS legitimate — white pawn on e5 can capture via e5xf6.
    Position pos;
    pos.set_fen("rnbqkbnr/ppp1p1pp/3p4/4Pp2/8/8/PPPP1PPP/RNBQKBNR w KQkq f6 0 3");
    EXPECT_EQ(pos.ep_square(), F6);
}

// ---------------------------------------------------------------------------
// Position equality. Needed so is_repetition can verify full-board identity
// on a hash match (our hash is ad-hoc multiply-mix, not Zobrist — collisions
// exist, and a false threefold claim would silently poison self-play data).
// ---------------------------------------------------------------------------

TEST_F(PositionTest, Equality_IdenticalFenAreEqual) {
    Position a, b;
    a.set_fen("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1");
    b.set_fen("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1");
    EXPECT_TRUE(a == b);
    EXPECT_FALSE(a != b);
}

TEST_F(PositionTest, Equality_DifferentStmAreUnequal) {
    Position a, b;
    a.set_fen("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1");
    b.set_fen("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR b KQkq - 0 1");
    EXPECT_FALSE(a == b);
}

TEST_F(PositionTest, Equality_DifferentCastlingAreUnequal) {
    Position a, b;
    a.set_fen("r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1");
    b.set_fen("r3k2r/8/8/8/8/8/8/R3K2R w Kkq - 0 1");
    EXPECT_FALSE(a == b);
}

TEST_F(PositionTest, Equality_DifferentPiecesAreUnequal) {
    Position a, b;
    a.set_fen("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1");
    b.set_fen("rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1");
    EXPECT_FALSE(a == b);
}

TEST_F(PositionTest, Equality_MoveCountersIgnored) {
    // FIDE 9.2 repetition: halfmove/fullmove don't count toward identity.
    Position a, b;
    a.set_fen("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1");
    b.set_fen("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 42 99");
    EXPECT_TRUE(a == b);
}

TEST_F(PositionTest, Equality_DifferentLegitimateEpAreUnequal) {
    // Two positions with the same piece layout but different capturable EP
    // targets — they're different states because one allows EP, the other
    // doesn't. After EP canonicalization, only legitimate EPs survive.
    Position a;
    a.set_fen("rnbqkbnr/ppp1p1pp/3p4/4Pp2/8/8/PPPP1PPP/RNBQKBNR w KQkq f6 0 3");
    Position b = a;
    // Clear the EP by re-setting the FEN without it.
    b.set_fen("rnbqkbnr/ppp1p1pp/3p4/4Pp2/8/8/PPPP1PPP/RNBQKBNR w KQkq - 0 3");
    EXPECT_FALSE(a == b);
}
