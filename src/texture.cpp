#include "keel/image.h"

#include "keel/arena.h"
#include "keel/array.h"
#include "keel/math.h"

namespace keel::image {

ImageStatus graycomatrix(ArrayRef<f64> img_col_major, usize height, usize width, i32 num_levels,
                         i32 off_r, i32 off_c, f64 g_low, f64 g_high, bool symmetric,
                         MutArrayRef<f64> out_glcm_col_major, Allocator* alloc) {
    if (num_levels < 2 || !(g_high > g_low)) {
        return ImageStatus::kInvalidArgument;
    }
    const usize npix = height * width;
    const usize nl = as<usize>(num_levels);
    if (img_col_major.size() < npix || out_glcm_col_major.size() < nl * nl) {
        return ImageStatus::kInvalidArgument;
    }

    for (usize i = 0; i < nl * nl; ++i) out_glcm_col_major[i] = 0.0;
    if (npix == 0u) return ImageStatus::kOk;

    ArenaTemp scratch = get_scratch(alloc);
    Array<i32> q(npix, -1, scratch.allocator());
    const f64 scale = as<f64>(num_levels) / (g_high - g_low);
    for (usize i = 0; i < npix; ++i) {
        const f64 v = img_col_major[i];
        if (keel::is_nan(v)) {
            q[i] = -1;
            continue;
        }
        i32 lvl = as<i32>(keel::floor((v - g_low) * scale));
        if (lvl < 0) lvl = 0;
        if (lvl >= num_levels) lvl = num_levels - 1;
        q[i] = lvl;
    }

    for (usize c = 0; c < width; ++c) {
        const i64 c2 = as<i64>(c) + as<i64>(off_c);
        if (c2 < 0 || c2 >= as<i64>(width)) continue;
        for (usize r = 0; r < height; ++r) {
            const i64 r2 = as<i64>(r) + as<i64>(off_r);
            if (r2 < 0 || r2 >= as<i64>(height)) continue;
            const i32 i_lvl = q[r + c * height];
            const i32 j_lvl = q[as<usize>(r2) + as<usize>(c2) * height];
            if (i_lvl < 0 || j_lvl < 0) continue;
            out_glcm_col_major[as<usize>(i_lvl) + as<usize>(j_lvl) * nl] += 1.0;
            if (symmetric) {
                out_glcm_col_major[as<usize>(j_lvl) + as<usize>(i_lvl) * nl] += 1.0;
            }
        }
    }
    return ImageStatus::kOk;
}

ImageStatus graycoprops(ArrayRef<f64> glcm_col_major, usize num_levels, GlcmProps& out) noexcept {
    if (num_levels == 0u || glcm_col_major.size() < num_levels * num_levels) {
        return ImageStatus::kInvalidArgument;
    }
    f64 total = 0.0;
    for (usize i = 0; i < num_levels * num_levels; ++i) total += glcm_col_major[i];
    if (!(total > 0.0)) return ImageStatus::kInvalidArgument;

    f64 mu_i = 0.0;
    f64 mu_j = 0.0;
    for (usize j = 0; j < num_levels; ++j) {
        for (usize i = 0; i < num_levels; ++i) {
            const f64 p = glcm_col_major[i + j * num_levels] / total;
            mu_i += as<f64>(i + 1u) * p;
            mu_j += as<f64>(j + 1u) * p;
        }
    }

    f64 var_i = 0.0;
    f64 var_j = 0.0;
    for (usize j = 0; j < num_levels; ++j) {
        for (usize i = 0; i < num_levels; ++i) {
            const f64 p = glcm_col_major[i + j * num_levels] / total;
            const f64 di = as<f64>(i + 1u) - mu_i;
            const f64 dj = as<f64>(j + 1u) - mu_j;
            var_i += di * di * p;
            var_j += dj * dj * p;
        }
    }
    const f64 sigma_i = keel::sqrt(var_i);
    const f64 sigma_j = keel::sqrt(var_j);

    f64 contrast = 0.0;
    f64 energy = 0.0;
    f64 homogeneity = 0.0;
    f64 correlation = 0.0;
    for (usize j = 0; j < num_levels; ++j) {
        for (usize i = 0; i < num_levels; ++i) {
            const f64 p = glcm_col_major[i + j * num_levels] / total;
            const f64 di = as<f64>(i) - as<f64>(j);
            contrast += di * di * p;
            energy += p * p;
            homogeneity += p / (1.0 + keel::abs(di));
            if (sigma_i > 0.0 && sigma_j > 0.0) {
                correlation +=
                    ((as<f64>(i + 1u) - mu_i) * (as<f64>(j + 1u) - mu_j) * p) / (sigma_i * sigma_j);
            }
        }
    }
    if (!(sigma_i > 0.0 && sigma_j > 0.0)) {
        correlation = Limits<f64>::quiet_nan();
    }

    out.contrast = contrast;
    out.correlation = correlation;
    out.energy = energy;
    out.homogeneity = homogeneity;
    return ImageStatus::kOk;
}

} // namespace keel::image
