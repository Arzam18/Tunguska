#include "Engine/Board.h"
#include "NNUE/nnue_loader.h"
#include "NNUE/FeatureExtractor.h"
#include <iostream>
#include <algorithm>
#include <cstring>
#include <cstdint>

#if defined(__AVX2__)
#include <immintrin.h>
#elif defined(__ARM_NEON) || defined(__ARM_NEON__)
#include <arm_neon.h>
#endif

// The network is embedded directly in this translation unit.
#include "NNUE/new_uho_34.h"

NNUEState::NNUEState()
    : accumulators(new NNUEAccumulator[Board::MAX_DEPTH]) {}

namespace {

inline void addFeature(int16_t* accumulator, const int16_t* weights) {
#if defined(__AVX2__)
    for (int j = 0; j < HL_SIZE; j += 16) {
        const __m256i a = _mm256_load_si256(reinterpret_cast<const __m256i*>(accumulator + j));
        const __m256i w = _mm256_load_si256(reinterpret_cast<const __m256i*>(weights + j));
        _mm256_store_si256(reinterpret_cast<__m256i*>(accumulator + j), _mm256_add_epi16(a, w));
    }
#elif defined(__ARM_NEON) || defined(__ARM_NEON__)
    for (int j = 0; j < HL_SIZE; j += 8) {
        const int16x8_t a = vld1q_s16(accumulator + j);
        const int16x8_t w = vld1q_s16(weights + j);
        vst1q_s16(accumulator + j, vaddq_s16(a, w));
    }
#else
    for (int j = 0; j < HL_SIZE; ++j)
        accumulator[j] = static_cast<int16_t>(accumulator[j] + weights[j]);
#endif
}

#if defined(__AVX2__)
template<int Removed, int Added>
void updateFused(const int16_t* parent, int16_t* child, const int* removed, const int* added) {
    const int16_t* remWeights[Removed];
    const int16_t* addWeights[Added];
    for (int i = 0; i < Removed; ++i)
        remWeights[i] = NNUENetwork::accumulator_weight[removed[i]];
    for (int i = 0; i < Added; ++i)
        addWeights[i] = NNUENetwork::accumulator_weight[added[i]];

    for (int j = 0; j < HL_SIZE; j += 16) {
        __m256i value = _mm256_load_si256(reinterpret_cast<const __m256i*>(parent + j));
        for (int i = 0; i < Removed; ++i) {
            const __m256i weight = _mm256_load_si256(reinterpret_cast<const __m256i*>(remWeights[i] + j));
            value = _mm256_sub_epi16(value, weight);
        }
        for (int i = 0; i < Added; ++i) {
            const __m256i weight = _mm256_load_si256(reinterpret_cast<const __m256i*>(addWeights[i] + j));
            value = _mm256_add_epi16(value, weight);
        }
        _mm256_store_si256(reinterpret_cast<__m256i*>(child + j), value);
    }
}
#elif defined(__ARM_NEON) || defined(__ARM_NEON__)
template<int Removed, int Added>
void updateFused(const int16_t* parent, int16_t* child, const int* removed, const int* added) {
    const int16_t* remWeights[Removed];
    const int16_t* addWeights[Added];
    for (int i = 0; i < Removed; ++i)
        remWeights[i] = NNUENetwork::accumulator_weight[removed[i]];
    for (int i = 0; i < Added; ++i)
        addWeights[i] = NNUENetwork::accumulator_weight[added[i]];

    for (int j = 0; j < HL_SIZE; j += 8) {
        int16x8_t value = vld1q_s16(parent + j);
        for (int i = 0; i < Removed; ++i)
            value = vsubq_s16(value, vld1q_s16(remWeights[i] + j));
        for (int i = 0; i < Added; ++i)
            value = vaddq_s16(value, vld1q_s16(addWeights[i] + j));
        vst1q_s16(child + j, value);
    }
}
#else
template<int Removed, int Added>
void updateFused(const int16_t* parent, int16_t* child, const int* removed, const int* added) {
    const int16_t* remWeights[Removed];
    const int16_t* addWeights[Added];
    for (int i = 0; i < Removed; ++i)
        remWeights[i] = NNUENetwork::accumulator_weight[removed[i]];
    for (int i = 0; i < Added; ++i)
        addWeights[i] = NNUENetwork::accumulator_weight[added[i]];
    for (int j = 0; j < HL_SIZE; ++j) {
        int value = parent[j];
        for (int i = 0; i < Removed; ++i) value -= remWeights[i][j];
        for (int i = 0; i < Added; ++i) value += addWeights[i][j];
        child[j] = static_cast<int16_t>(value);
    }
}
#endif

void updateDynamic(const int16_t* parent, int16_t* child, const int* removed, int removedCount,
                   const int* added, int addedCount) {
    const int16_t* remWeights[4];
    const int16_t* addWeights[4];
    for (int i = 0; i < removedCount; ++i)
        remWeights[i] = NNUENetwork::accumulator_weight[removed[i]];
    for (int i = 0; i < addedCount; ++i)
        addWeights[i] = NNUENetwork::accumulator_weight[added[i]];

#if defined(__AVX2__)
    for (int j = 0; j < HL_SIZE; j += 16) {
        __m256i value = _mm256_load_si256(reinterpret_cast<const __m256i*>(parent + j));
        for (int i = 0; i < removedCount; ++i)
            value = _mm256_sub_epi16(value, _mm256_load_si256(reinterpret_cast<const __m256i*>(remWeights[i] + j)));
        for (int i = 0; i < addedCount; ++i)
            value = _mm256_add_epi16(value, _mm256_load_si256(reinterpret_cast<const __m256i*>(addWeights[i] + j)));
        _mm256_store_si256(reinterpret_cast<__m256i*>(child + j), value);
    }
#elif defined(__ARM_NEON) || defined(__ARM_NEON__)
    for (int j = 0; j < HL_SIZE; j += 8) {
        int16x8_t value = vld1q_s16(parent + j);
        for (int i = 0; i < removedCount; ++i)
            value = vsubq_s16(value, vld1q_s16(remWeights[i] + j));
        for (int i = 0; i < addedCount; ++i)
            value = vaddq_s16(value, vld1q_s16(addWeights[i] + j));
        vst1q_s16(child + j, value);
    }
#else
    for (int j = 0; j < HL_SIZE; ++j) {
        int value = parent[j];
        for (int i = 0; i < removedCount; ++i) value -= remWeights[i][j];
        for (int i = 0; i < addedCount; ++i) value += addWeights[i][j];
        child[j] = static_cast<int16_t>(value);
    }
#endif
}

#if defined(__ARM_FEATURE_DOTPROD) && (defined(__ARM_NEON) || defined(__ARM_NEON__))
// Keep the Dotprod instruction explicit. This avoids depending on whether a
// particular compiler version lowers the ACLE intrinsic during optimization.
static inline int32x4_t dotprodAccumulate(int32x4_t accumulator,
                                          int8x16_t lhs,
                                          int8x16_t rhs) {
    __asm__ volatile("sdot %0.4s, %1.16b, %2.16b"
                     : "+w"(accumulator)
                     : "w"(lhs), "w"(rhs));
    return accumulator;
}
#endif

void updatePerspective(const int16_t* parent, int16_t* child, const int* removed, int removedCount,
                       const int* added, int addedCount) {
    if (removedCount == 1 && addedCount == 1) {
        updateFused<1, 1>(parent, child, removed, added);
    } else if (removedCount == 2 && addedCount == 1) {
        updateFused<2, 1>(parent, child, removed, added);
    } else if (removedCount == 2 && addedCount == 2) {
        updateFused<2, 2>(parent, child, removed, added);
    } else {
        // Includes empty deltas: the child stack slot must still receive a copy.
        updateDynamic(parent, child, removed, removedCount, added, addedCount);
    }
}

} // namespace

void NNUEState::init(const Board& board) {
    accumulatorIndex = 0;
    if (!NNUENetwork::weights_loaded) return;
    NNUEAccumulator& accumulator = accumulators[0];

    StartingFeatures initialFeatures;
    FeatureExtractor::extractFeatures(board, initialFeatures);

    std::memcpy(accumulator.v[WHITE_NNUE], NNUENetwork::accumulator_bias, HL_SIZE * sizeof(int16_t));
    std::memcpy(accumulator.v[BLACK_NNUE], NNUENetwork::accumulator_bias, HL_SIZE * sizeof(int16_t));

    for (int i = 0; i < initialFeatures.white_feats_cnt; ++i) {
        const int feat = initialFeatures.white_feats_idx[i];
        addFeature(accumulator.v[WHITE_NNUE], NNUENetwork::accumulator_weight[feat]);
    }
    for (int i = 0; i < initialFeatures.black_feats_cnt; ++i) {
        const int feat = initialFeatures.black_feats_idx[i];
        addFeature(accumulator.v[BLACK_NNUE], NNUENetwork::accumulator_weight[feat]);
    }
}

void NNUEState::update(const FeatureChanges& changes) {
    assert(accumulatorIndex + 1 < Board::MAX_DEPTH);
    const NNUEAccumulator& parent = accumulators[accumulatorIndex];
    NNUEAccumulator& child = accumulators[accumulatorIndex + 1];
    updatePerspective(parent.v[WHITE_NNUE], child.v[WHITE_NNUE], changes.rem_white, changes.rem_white_count,
                      changes.add_white, changes.add_white_count);
    updatePerspective(parent.v[BLACK_NNUE], child.v[BLACK_NNUE], changes.rem_black, changes.rem_black_count,
                      changes.add_black, changes.add_black_count);
    ++accumulatorIndex;
}

int NNUEState::evaluate(Side stm) {
    const Side them = static_cast<Side>(1 - stm);
    const NNUEAccumulator& accumulator = currentAccumulator();
    const int16_t* us_acc = accumulator.v[stm];
    const int16_t* them_acc = accumulator.v[them];
    const int16_t* out_w_us = NNUENetwork::output_weights;
    const int16_t* out_w_them = NNUENetwork::output_weights + HL_SIZE;

#if defined(__AVX2__)
    const __m256i zero = _mm256_setzero_si256();
    const __m256i qa = _mm256_set1_epi16(QA);
    __m256i sum256 = _mm256_setzero_si256();

    for (int i = 0; i < HL_SIZE; i += 16) {
        __m256i act_us = _mm256_load_si256(reinterpret_cast<const __m256i*>(us_acc + i));
        act_us = _mm256_max_epi16(act_us, zero);
        act_us = _mm256_min_epi16(act_us, qa);
        const __m256i w_us = _mm256_load_si256(reinterpret_cast<const __m256i*>(out_w_us + i));
        sum256 = _mm256_add_epi32(sum256, _mm256_madd_epi16(act_us, w_us));

        __m256i act_them = _mm256_load_si256(reinterpret_cast<const __m256i*>(them_acc + i));
        act_them = _mm256_max_epi16(act_them, zero);
        act_them = _mm256_min_epi16(act_them, qa);
        const __m256i w_them = _mm256_load_si256(reinterpret_cast<const __m256i*>(out_w_them + i));
        sum256 = _mm256_add_epi32(sum256, _mm256_madd_epi16(act_them, w_them));
    }

    __m128i sum128 = _mm_add_epi32(_mm256_castsi256_si128(sum256), _mm256_extracti128_si256(sum256, 1));
    sum128 = _mm_add_epi32(sum128, _mm_shuffle_epi32(sum128, _MM_SHUFFLE(1, 0, 3, 2)));
    sum128 = _mm_add_epi32(sum128, _mm_shuffle_epi32(sum128, _MM_SHUFFLE(2, 3, 0, 1)));
    const int final_sum = _mm_cvtsi128_si32(sum128);
    return (final_sum + NNUENetwork::output_bias) / QA;

#elif defined(__ARM_FEATURE_DOTPROD) && (defined(__ARM_NEON) || defined(__ARM_NEON__))
    // Exact SDOT path for the network's int16 output weights.
    // Each weight W is decomposed as W = low_signed + 128 + 256 * high_signed,
    // where low_signed = (W & 0xff) - 128 and high_signed = W >> 8.
    // Activations are centered by 128. The correction terms below restore the
    // exact original int16 dot product; no network quantization is performed.
    const int16x8_t zero16 = vdupq_n_s16(0);
    const int16x8_t qa16 = vdupq_n_s16(QA);
    const int16x8_t offset16 = vdupq_n_s16(128);
    const uint8x8_t byteSign = vdup_n_u8(0x80);
    int32x4_t dotLow = vdupq_n_s32(0);
    int32x4_t dotHigh = vdupq_n_s32(0);
    int64_t sumCenteredActivations = 0;
    int64_t sumWeights = 0;

    for (int i = 0; i < HL_SIZE; i += 16) {
        const int16x8_t us0 = vminq_s16(vmaxq_s16(vld1q_s16(us_acc + i), zero16), qa16);
        const int16x8_t us1 = vminq_s16(vmaxq_s16(vld1q_s16(us_acc + i + 8), zero16), qa16);
        const int16x8_t th0 = vminq_s16(vmaxq_s16(vld1q_s16(them_acc + i), zero16), qa16);
        const int16x8_t th1 = vminq_s16(vmaxq_s16(vld1q_s16(them_acc + i + 8), zero16), qa16);

        const int16x8_t wu0 = vld1q_s16(out_w_us + i);
        const int16x8_t wu1 = vld1q_s16(out_w_us + i + 8);
        const int16x8_t wt0 = vld1q_s16(out_w_them + i);
        const int16x8_t wt1 = vld1q_s16(out_w_them + i + 8);

        const int8x16_t au = vcombine_s8(vmovn_s16(vsubq_s16(us0, offset16)),
                                         vmovn_s16(vsubq_s16(us1, offset16)));
        const int8x16_t at = vcombine_s8(vmovn_s16(vsubq_s16(th0, offset16)),
                                         vmovn_s16(vsubq_s16(th1, offset16)));

        const int8x16_t wuLow = vcombine_s8(
            vreinterpret_s8_u8(veor_u8(vmovn_u16(vreinterpretq_u16_s16(wu0)), byteSign)),
            vreinterpret_s8_u8(veor_u8(vmovn_u16(vreinterpretq_u16_s16(wu1)), byteSign)));
        const int8x16_t wtLow = vcombine_s8(
            vreinterpret_s8_u8(veor_u8(vmovn_u16(vreinterpretq_u16_s16(wt0)), byteSign)),
            vreinterpret_s8_u8(veor_u8(vmovn_u16(vreinterpretq_u16_s16(wt1)), byteSign)));

        const int8x16_t wuHigh = vcombine_s8(vmovn_s16(vshrq_n_s16(wu0, 8)),
                                             vmovn_s16(vshrq_n_s16(wu1, 8)));
        const int8x16_t wtHigh = vcombine_s8(vmovn_s16(vshrq_n_s16(wt0, 8)),
                                             vmovn_s16(vshrq_n_s16(wt1, 8)));

        dotLow = dotprodAccumulate(dotLow, au, wuLow);
        dotLow = dotprodAccumulate(dotLow, at, wtLow);
        dotHigh = dotprodAccumulate(dotHigh, au, wuHigh);
        dotHigh = dotprodAccumulate(dotHigh, at, wtHigh);

        sumCenteredActivations += vaddlvq_s16(vsubq_s16(us0, offset16));
        sumCenteredActivations += vaddlvq_s16(vsubq_s16(us1, offset16));
        sumCenteredActivations += vaddlvq_s16(vsubq_s16(th0, offset16));
        sumCenteredActivations += vaddlvq_s16(vsubq_s16(th1, offset16));
        sumWeights += vaddlvq_s16(wu0) + vaddlvq_s16(wu1);
        sumWeights += vaddlvq_s16(wt0) + vaddlvq_s16(wt1);
    }

    const int64_t finalSum = static_cast<int64_t>(vaddvq_s32(dotLow))
                           + 256LL * static_cast<int64_t>(vaddvq_s32(dotHigh))
                           + 128LL * (sumCenteredActivations + sumWeights);
    return static_cast<int>((finalSum + NNUENetwork::output_bias) / QA);

#elif defined(__ARM_NEON) || defined(__ARM_NEON__)
    const int16x8_t zero = vdupq_n_s16(0);
    const int16x8_t qa = vdupq_n_s16(QA);
    int32x4_t sum = vdupq_n_s32(0);

    for (int i = 0; i < HL_SIZE; i += 8) {
        int16x8_t actUs = vminq_s16(vmaxq_s16(vld1q_s16(us_acc + i), zero), qa);
        int16x8_t actThem = vminq_s16(vmaxq_s16(vld1q_s16(them_acc + i), zero), qa);
        const int16x8_t wUs = vld1q_s16(out_w_us + i);
        const int16x8_t wThem = vld1q_s16(out_w_them + i);

        sum = vaddq_s32(sum, vmull_s16(vget_low_s16(actUs), vget_low_s16(wUs)));
        sum = vaddq_s32(sum, vmull_s16(vget_high_s16(actUs), vget_high_s16(wUs)));
        sum = vaddq_s32(sum, vmull_s16(vget_low_s16(actThem), vget_low_s16(wThem)));
        sum = vaddq_s32(sum, vmull_s16(vget_high_s16(actThem), vget_high_s16(wThem)));
    }

    const int final_sum = vaddvq_s32(sum);
    return (final_sum + NNUENetwork::output_bias) / QA;

#else
    int64_t final_sum = 0;
    for (int i = 0; i < HL_SIZE; ++i) {
        const int us = std::max(0, std::min(QA, static_cast<int>(us_acc[i])));
        const int other = std::max(0, std::min(QA, static_cast<int>(them_acc[i])));
        final_sum += static_cast<int64_t>(us) * out_w_us[i];
        final_sum += static_cast<int64_t>(other) * out_w_them[i];
    }
    return static_cast<int>((final_sum + NNUENetwork::output_bias) / QA);
#endif
}
