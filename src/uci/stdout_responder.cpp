#include "uci/stdout_responder.h"
#include <sstream>

namespace uci {

namespace {

// Render score as "mate N" (full moves) or "cp X". Stockfish convention:
// plies → moves via ceiling for positive (e.g. mate-in-3 plies = "mate 2"),
// floor for negative. See engines/stockfish/src/uci.cpp::format_score.
std::string format_score(const ThinkingInfo& info) {
    if (info.mate_plies) {
        int plies = *info.mate_plies;
        int full_moves = (plies > 0 ? (plies + 1) : plies) / 2;
        return "mate " + std::to_string(full_moves);
    }
    if (info.score_cp) {
        return "cp " + std::to_string(*info.score_cp);
    }
    return {};
}

std::string build_info_line(const ThinkingInfo& info) {
    std::ostringstream ss;
    ss << "info";
    if (info.multipv)       ss << " multipv "  << *info.multipv;
    if (info.depth > 0)     ss << " depth "    << info.depth;
    if (info.seldepth)      ss << " seldepth " << *info.seldepth;
    if (info.nodes > 0)     ss << " nodes "    << info.nodes;
    if (info.nps > 0)       ss << " nps "      << info.nps;
    if (info.time_ms >= 0)  ss << " time "     << info.time_ms;

    const std::string score = format_score(info);
    if (!score.empty())     ss << " score "    << score;
    if (info.wdl)           ss << " wdl "      << info.wdl->w << " "
                                                << info.wdl->d << " "
                                                << info.wdl->l;
    if (info.hashfull)      ss << " hashfull " << *info.hashfull;
    if (info.tbhits)        ss << " tbhits "   << *info.tbhits;

    if (!info.pv.empty()) {
        ss << " pv";
        for (const auto& m : info.pv) ss << " " << m;
    }
    return ss.str();
}

}  // namespace

StdoutResponder::StdoutResponder(std::ostream& out) : out_(out) {}

void StdoutResponder::OnInfo(const ThinkingInfo& info) {
    std::lock_guard<std::mutex> lk(mutex_);
    if (!info.comment.empty()) {
        out_ << "info string " << info.comment << "\n";
    }
    out_ << build_info_line(info) << "\n";
    out_.flush();
}

void StdoutResponder::OnBestMove(const BestMoveInfo& info) {
    std::lock_guard<std::mutex> lk(mutex_);
    out_ << "bestmove " << info.bestmove;
    if (info.ponder && !info.ponder->empty()) {
        out_ << " ponder " << *info.ponder;
    }
    out_ << "\n";
    out_.flush();
}

void StdoutResponder::OnString(const std::string& msg) {
    std::lock_guard<std::mutex> lk(mutex_);
    out_ << "info string " << msg << "\n";
    out_.flush();
}

void StdoutResponder::OnRaw(const std::string& line) {
    std::lock_guard<std::mutex> lk(mutex_);
    out_ << line << "\n";
    out_.flush();
}

}  // namespace uci
