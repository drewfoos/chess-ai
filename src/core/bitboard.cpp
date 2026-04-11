#include "core/bitboard.h"
#include <sstream>

std::string bitboard_to_string(Bitboard bb) {
    std::ostringstream ss;
    for (int rank = 7; rank >= 0; --rank) {
        for (int file = 0; file < 8; ++file) {
            Square s = make_square(file, rank);
            ss << ((bb & square_bb(s)) ? '1' : '.') << ' ';
        }
        ss << '\n';
    }
    return ss.str();
}
