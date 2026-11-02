#pragma once

#include <algorithm>
#include <charconv>
#include <cstdint>
#include <string>
#include <utility>
#include <vector>

namespace strata::core {

inline constexpr int kThreeLevelMaxCandidates = 24;
inline constexpr int kThreeLevelMaxRows = kThreeLevelMaxCandidates + 1;

enum class SpeculativeStage { PreVerify, FullVerify };

struct ThreeLevelConfig {
    bool enabled = false;
    int depth = 3;
    int loops = 6;
    int top_h = 4;

    int max_candidates() const { return (depth + 1) * loops; }
    bool validate(std::string& err, int router_k = 10) const {
        if (depth < 1 || depth >= kThreeLevelMaxCandidates || loops < 1 ||
            loops > kThreeLevelMaxCandidates / (depth + 1)) {
            err = "three-level: depth and loops must be positive and (depth + 1) * loops must be <= 24";
            return false;
        }
        if (top_h < 1 || top_h > router_k) {
            err = "three-level: preverify-top-h must be in 1.." + std::to_string(router_k);
            return false;
        }
        return true;
    }
};

inline bool three_level_parse_int(const std::string& value, int& out) {
    int parsed = 0;
    const auto r = std::from_chars(value.data(), value.data() + value.size(), parsed);
    if (r.ec != std::errc{} || r.ptr != value.data() + value.size()) return false;
    out = parsed;
    return true;
}

struct ThreeLevelResult {
    std::vector<int32_t> candidates;
    std::vector<int32_t> outputs;
    int32_t next_anchor = 0;
    int accepted_candidates = 0;
    int committed_inputs = 0;
    int preverify_calls = 0;
    int preverify_inputs = 0;
    int preverify_drafted = 0;
    int preverify_accepted = 0;
    int full_verify_inputs = 0;
    bool stopped_eos = false;
    bool cancelled = false;
};

// The adapter owns native state, graphs and dispatch. The controller never publishes private predictions or
// consumes the last output: that output is the next window's already-confirmed, unconsumed anchor.
// begin_window saves confirmed state and selects a private branch; select_confirmed restores it before Full.
// verify is uncommitted; commit advances exactly its first n input rows and catches MTP up to that boundary.
// abort_window must restore the saved base even after a partially failed commit; finish_window keeps Full's commit.
template <class Adapter>
bool run_three_level_window(Adapter& adapter, const ThreeLevelConfig& cfg, int32_t anchor,
                            int64_t output_budget, int64_t context_inputs, ThreeLevelResult& result,
                            std::string& err) {
    result = {};
    result.next_anchor = anchor;
    if (!cfg.validate(err)) return false;
    if (output_budget <= 0 || context_inputs <= 0) return true;
    if (adapter.cancelled()) { result.cancelled = true; return true; }

    struct BranchGuard {
        Adapter& adapter;
        bool finished = false;
        ~BranchGuard() { if (!finished) adapter.abort_window(); }
    } guard{adapter};
    if (!adapter.begin_window(err)) return false;

    const int capacity = (int) std::min<int64_t>(cfg.max_candidates(),
        std::min(output_budget - 1, context_inputs - 1));
    std::vector<int32_t> drafts, inputs, picks;
    int32_t private_anchor = anchor;
    for (int loop = 0; loop < cfg.loops && (int) result.candidates.size() < capacity; ++loop) {
        if (adapter.cancelled()) { result.cancelled = true; return true; }
        const int room = capacity - (int) result.candidates.size();
        const int depth = std::min(cfg.depth, room - 1);
        drafts.clear();
        if (!adapter.draft(private_anchor, depth, drafts, err)) return false;
        if ((int) drafts.size() != depth) {
            err = "three-level: drafter returned a different number of tokens than requested";
            return false;
        }
        if (adapter.cancelled()) { result.cancelled = true; return true; }
        inputs.assign(1, private_anchor);
        inputs.insert(inputs.end(), drafts.begin(), drafts.end());
        picks.clear();
        if (!adapter.verify(SpeculativeStage::PreVerify, inputs, picks, err)) return false;
        if (picks.size() != inputs.size()) {
            err = "three-level: Pre-Verify must return one greedy pick per input row";
            return false;
        }
        ++result.preverify_calls;
        result.preverify_inputs += (int) inputs.size();
        result.preverify_drafted += depth;
        int accepted = 0;
        while (accepted < depth && drafts[(size_t) accepted] == picks[(size_t) accepted]) ++accepted;
        int keep = accepted + 1;
        bool eos = false;
        for (int i = 0; i < keep; ++i)
            if (adapter.is_eos(picks[(size_t) i])) { keep = i + 1; eos = true; break; }
        result.preverify_accepted += std::min(accepted, keep);
        if (adapter.cancelled()) { result.cancelled = true; return true; }
        if (!adapter.commit(SpeculativeStage::PreVerify, keep, err)) return false;
        result.candidates.insert(result.candidates.end(), picks.begin(), picks.begin() + keep);
        private_anchor = picks[(size_t) keep - 1];
        if (eos) break;
    }
    if (adapter.cancelled()) { result.cancelled = true; return true; }
    if (!adapter.select_confirmed(err)) return false;
    inputs.assign(1, anchor);
    inputs.insert(inputs.end(), result.candidates.begin(), result.candidates.end());
    result.full_verify_inputs = (int) inputs.size();
    picks.clear();
    if (!adapter.verify(SpeculativeStage::FullVerify, inputs, picks, err)) return false;
    if (picks.size() != inputs.size()) {
        err = "three-level: Full Verify must return one greedy pick per input row";
        return false;
    }
    int accepted = 0;
    while (accepted < (int) result.candidates.size() &&
           result.candidates[(size_t) accepted] == picks[(size_t) accepted]) ++accepted;
    int keep = accepted + 1;
    for (int i = 0; i < keep; ++i)
        if (adapter.is_eos(picks[(size_t) i])) { keep = i + 1; result.stopped_eos = true; break; }
    if (adapter.cancelled()) { result.cancelled = true; return true; }
    if (!adapter.commit(SpeculativeStage::FullVerify, keep, err)) return false;
    if (adapter.cancelled()) { result.cancelled = true; return true; }
    std::vector<int32_t> confirmed(picks.begin(), picks.begin() + keep);
    adapter.finish_window();
    guard.finished = true;
    result.accepted_candidates = std::min(accepted, keep);
    result.committed_inputs = keep;
    result.outputs = std::move(confirmed);
    result.next_anchor = result.outputs.back();
    return true;
}

}  // namespace strata::core
