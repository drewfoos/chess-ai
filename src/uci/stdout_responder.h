#pragma once
#include "uci/responder.h"
#include <iostream>
#include <mutex>

namespace uci {

// Default Responder implementation. Formats ThinkingInfo/BestMoveInfo into
// UCI text and writes to the provided ostream (std::cout in production,
// a stringstream in tests). All four OnX methods hold the same mutex so
// that a concurrent info line from the search thread can't interleave with
// a handshake response from the main thread.
class StdoutResponder : public Responder {
public:
    explicit StdoutResponder(std::ostream& out = std::cout);

    void OnInfo(const ThinkingInfo& info) override;
    void OnBestMove(const BestMoveInfo& info) override;
    void OnString(const std::string& msg) override;
    void OnRaw(const std::string& line) override;

private:
    std::ostream& out_;
    std::mutex mutex_;
};

}  // namespace uci
