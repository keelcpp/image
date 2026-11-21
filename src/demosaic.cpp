#include "keel/image.h"

namespace keel::image {

namespace {

constexpr f64 kGAtRb[25] = {
    0.0, 0.0, -1.0, 0.0, 0.0,
    0.0, 0.0,  2.0, 0.0, 0.0,
   -1.0, 2.0,  4.0, 2.0, -1.0,
    0.0, 0.0,  2.0, 0.0, 0.0,
    0.0, 0.0, -1.0, 0.0, 0.0,
};

constexpr f64 kCAtGSameRow[25] = {
    0.0,  0.0, 0.5,  0.0,  0.0,
    0.0, -1.0, 0.0, -1.0,  0.0,
   -1.0,  4.0, 5.0,  4.0, -1.0,
    0.0, -1.0, 0.0, -1.0,  0.0,
    0.0,  0.0, 0.5,  0.0,  0.0,
};

constexpr f64 kCAtGDiffRow[25] = {
    0.0,  0.0, -1.0,  0.0, 0.0,
    0.0, -1.0,  4.0, -1.0, 0.0,
    0.5,  0.0,  5.0,  0.0, 0.5,
    0.0, -1.0,  4.0, -1.0, 0.0,
    0.0,  0.0, -1.0,  0.0, 0.0,
};

constexpr f64 kCAtC2[25] = {
    0.0,   0.0, -1.5, 0.0,  0.0,
    0.0,   2.0,  0.0, 2.0,  0.0,
   -1.5,   0.0,  6.0, 0.0, -1.5,
    0.0,   2.0,  0.0, 2.0,  0.0,
    0.0,   0.0, -1.5, 0.0,  0.0,
};

inline i32 mirror1(i32 k, i32 n) noexcept {
    if (k < 0) return -k;
    if (k >= n) return 2 * n - 2 - k;
    return k;
}

inline f64 sample_m(ArrayRef<f64> mosaic, i32 r, i32 c, usize m, usize n) noexcept {
    const i32 rp = mirror1(r, as<i32>(m));
    const i32 cp = mirror1(c, as<i32>(n));
    return mosaic[as<usize>(rp) + as<usize>(cp) * m];
}

inline f64 conv5(ArrayRef<f64> mosaic, i32 r, i32 c, usize m, usize n, const f64* k) noexcept {
    f64 acc = 0.0;
    for (i32 dr = -2; dr <= 2; ++dr) {
        for (i32 dc = -2; dc <= 2; ++dc) {
            const f64 w = k[(dr + 2) * 5 + (dc + 2)];
            if (w != 0.0) {
                acc += w * sample_m(mosaic, r + dr, c + dc, m, n);
            }
        }
    }
    return acc * 0.125;
}

inline f64 clamp_pixel(f64 v, f64 max_val) noexcept {
    if (v < 0.0) v = 0.0;
    if (max_val > 0.0 && v > max_val) v = max_val;
    return v;
}

} // namespace

ImageStatus demosaic_mhc(ArrayRef<f64> mosaic_col_major, usize height, usize width,
                         BayerPattern pattern, f64 max_val, MutArrayRef<f64> out_r,
                         MutArrayRef<f64> out_g, MutArrayRef<f64> out_b) noexcept {
    if (height < 2u || width < 2u || (height % 2u) != 0u || (width % 2u) != 0u) {
        return ImageStatus::kInvalidArgument;
    }
    const usize npix = height * width;
    if (mosaic_col_major.size() < npix || out_r.size() < npix || out_g.size() < npix ||
        out_b.size() < npix) {
        return ImageStatus::kInvalidArgument;
    }

    i32 r_par_r = 1;
    i32 r_par_c = 1;
    i32 b_par_r = 0;
    i32 b_par_c = 0;
    switch (pattern) {
        case BayerPattern::kRggb: r_par_r = 1; r_par_c = 1; b_par_r = 0; b_par_c = 0; break;
        case BayerPattern::kBggr: r_par_r = 0; r_par_c = 0; b_par_r = 1; b_par_c = 1; break;
        case BayerPattern::kGrbg: r_par_r = 1; r_par_c = 0; b_par_r = 0; b_par_c = 1; break;
        case BayerPattern::kGbrg: r_par_r = 0; r_par_c = 1; b_par_r = 1; b_par_c = 0; break;
    }

    for (i32 c = 0; c < as<i32>(width); ++c) {
        const i32 c1 = c + 1;
        for (i32 r = 0; r < as<i32>(height); ++r) {
            const i32 r1 = r + 1;
            const bool is_r = ((r1 & 1) == r_par_r) && ((c1 & 1) == r_par_c);
            const bool is_b = ((r1 & 1) == b_par_r) && ((c1 & 1) == b_par_c);

            f64 rv = 0.0;
            f64 gv = 0.0;
            f64 bv = 0.0;
            const f64 mv = sample_m(mosaic_col_major, r, c, height, width);
            if (is_r) {
                rv = mv;
                gv = conv5(mosaic_col_major, r, c, height, width, kGAtRb);
                bv = conv5(mosaic_col_major, r, c, height, width, kCAtC2);
            } else if (is_b) {
                bv = mv;
                gv = conv5(mosaic_col_major, r, c, height, width, kGAtRb);
                rv = conv5(mosaic_col_major, r, c, height, width, kCAtC2);
            } else {
                gv = mv;
                const bool g_in_r_row = ((r1 & 1) == r_par_r);
                if (g_in_r_row) {
                    rv = conv5(mosaic_col_major, r, c, height, width, kCAtGSameRow);
                    bv = conv5(mosaic_col_major, r, c, height, width, kCAtGDiffRow);
                } else {
                    rv = conv5(mosaic_col_major, r, c, height, width, kCAtGDiffRow);
                    bv = conv5(mosaic_col_major, r, c, height, width, kCAtGSameRow);
                }
            }

            const usize idx = as<usize>(r) + as<usize>(c) * height;
            out_r[idx] = clamp_pixel(rv, max_val);
            out_g[idx] = clamp_pixel(gv, max_val);
            out_b[idx] = clamp_pixel(bv, max_val);
        }
    }
    return ImageStatus::kOk;
}

} // namespace keel::image
