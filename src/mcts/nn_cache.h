#pragma once
#include <unordered_map>
#include <vector>
#include <cstdint>

namespace mcts {

struct CacheEntry {
    std::vector<float> policy;  // Per legal move priors (normalized)
    float value;                // Position value [-1, 1]
    int num_moves;              // Number of legal moves (policy.size())
};

class NNCache {
public:
    explicit NNCache(int max_size = 200000) : max_size_(max_size) {}

    // Returns pointer to cached entry, or nullptr if not found
    const CacheEntry* get(uint64_t hash) const {
        auto it = cache_.find(hash);
        return it != cache_.end() ? &it->second : nullptr;
    }

    // Store a new entry
    void put(uint64_t hash, CacheEntry entry) {
        if (static_cast<int>(cache_.size()) >= max_size_) {
            evict();
        }
        cache_[hash] = std::move(entry);
    }

    void clear() { cache_.clear(); }
    int size() const { return static_cast<int>(cache_.size()); }

private:
    std::unordered_map<uint64_t, CacheEntry> cache_;
    int max_size_;

    // Evict oldest 25% of entries
    void evict() {
        int to_remove = max_size_ / 4;
        auto it = cache_.begin();
        for (int i = 0; i < to_remove && it != cache_.end(); i++) {
            it = cache_.erase(it);
        }
    }
};

} // namespace mcts
