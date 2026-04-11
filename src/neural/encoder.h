#pragma once
#include "core/position.h"

namespace neural {

constexpr int INPUT_PLANES = 112;
constexpr int BOARD_SIZE = 8;
constexpr int TENSOR_SIZE = INPUT_PLANES * BOARD_SIZE * BOARD_SIZE;  // 7168

void encode_position(const Position& pos, float* output);

} // namespace neural
