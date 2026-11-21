#include "keel/image.h"

#include "keel/arena.h"
#include "keel/array.h"
#include "keel/math.h"

namespace keel::image {

namespace {

constexpr f64 kFrangiBeta = 0.5;
constexpr f64 kFrangiAlpha = 0.5;

inline usize clamp_idx(i64 i, usize n) noexcept {
    if (i < 0) return 0u;
    if (i >= as<i64>(n)) return n - 1u;
    return as<usize>(i);
}

Array<f64> gauss_1d(f64 sigma, i32 radius, Allocator* alloc) {
    const i32 n = 2 * radius + 1;
    Array<f64> k(as<usize>(n), 0.0, alloc);
    f64 s = 0.0;
    for (i32 i = 0; i < n; ++i) {
        const f64 x = as<f64>(i - radius);
        const f64 v = keel::exp(-0.5 * (x / sigma) * (x / sigma));
        k[as<usize>(i)] = v;
        s += v;
    }
    const f64 inv = 1.0 / s;
    for (usize i = 0; i < k.size(); ++i) k[i] *= inv;
    return k;
}

void gauss_2d_sep(ArrayRef<f64> in, usize h, usize w, f64 sigma, MutArrayRef<f64> out,
                  Allocator* alloc) {
    const i32 rad = as<i32>(keel::ceil(3.0 * sigma));
    const Array<f64> k = gauss_1d(sigma, rad, alloc);
    Array<f64> tmp(h * w, 0.0, alloc);

    for (usize c = 0; c < w; ++c) {
        for (usize r = 0; r < h; ++r) {
            f64 s = 0.0;
            for (i32 j = -rad; j <= rad; ++j) {
                const usize rr = clamp_idx(as<i64>(r) + as<i64>(j), h);
                s += k[as<usize>(j + rad)] * in[c * h + rr];
            }
            tmp[c * h + r] = s;
        }
    }
    for (usize c = 0; c < w; ++c) {
        for (usize r = 0; r < h; ++r) {
            f64 s = 0.0;
            for (i32 j = -rad; j <= rad; ++j) {
                const usize cc = clamp_idx(as<i64>(c) + as<i64>(j), w);
                s += k[as<usize>(j + rad)] * tmp[cc * h + r];
            }
            out[c * h + r] = s;
        }
    }
}

void gauss_3d_sep(ArrayRef<f64> in, usize h, usize w, usize d, f64 sigma, MutArrayRef<f64> out,
                  Allocator* alloc) {
    const i32 rad = as<i32>(keel::ceil(3.0 * sigma));
    const Array<f64> k = gauss_1d(sigma, rad, alloc);
    const usize hw = h * w;
    Array<f64> tmp1(hw * d, 0.0, alloc);
    Array<f64> tmp2(hw * d, 0.0, alloc);

    for (usize z = 0; z < d; ++z) {
        const usize zoff = z * hw;
        for (usize c = 0; c < w; ++c) {
            for (usize r = 0; r < h; ++r) {
                f64 s = 0.0;
                for (i32 j = -rad; j <= rad; ++j) {
                    const usize rr = clamp_idx(as<i64>(r) + as<i64>(j), h);
                    s += k[as<usize>(j + rad)] * in[zoff + c * h + rr];
                }
                tmp1[zoff + c * h + r] = s;
            }
        }
    }
    for (usize z = 0; z < d; ++z) {
        const usize zoff = z * hw;
        for (usize c = 0; c < w; ++c) {
            for (usize r = 0; r < h; ++r) {
                f64 s = 0.0;
                for (i32 j = -rad; j <= rad; ++j) {
                    const usize cc = clamp_idx(as<i64>(c) + as<i64>(j), w);
                    s += k[as<usize>(j + rad)] * tmp1[zoff + cc * h + r];
                }
                tmp2[zoff + c * h + r] = s;
            }
        }
    }
    for (usize z = 0; z < d; ++z) {
        const usize zoff = z * hw;
        for (usize c = 0; c < w; ++c) {
            for (usize r = 0; r < h; ++r) {
                f64 s = 0.0;
                for (i32 j = -rad; j <= rad; ++j) {
                    const usize zz = clamp_idx(as<i64>(z) + as<i64>(j), d);
                    s += k[as<usize>(j + rad)] * tmp2[zz * hw + c * h + r];
                }
                out[zoff + c * h + r] = s;
            }
        }
    }
}

void eigvals_3x3_sym(f64 a11, f64 a22, f64 a33, f64 a12, f64 a13, f64 a23, f64& l1, f64& l2,
                     f64& l3) noexcept {
    const f64 p1 = a12 * a12 + a13 * a13 + a23 * a23;
    if (p1 == 0.0) {
        l1 = a11;
        l2 = a22;
        l3 = a33;
    } else {
        const f64 q = (a11 + a22 + a33) / 3.0;
        const f64 p2 = (a11 - q) * (a11 - q) + (a22 - q) * (a22 - q) + (a33 - q) * (a33 - q) +
                       2.0 * p1;
        const f64 p = keel::sqrt(p2 / 6.0);
        if (p == 0.0) {
            l1 = l2 = l3 = q;
            return;
        }
        const f64 b11 = (a11 - q) / p;
        const f64 b22 = (a22 - q) / p;
        const f64 b33 = (a33 - q) / p;
        const f64 b12 = a12 / p;
        const f64 b13 = a13 / p;
        const f64 b23 = a23 / p;
        f64 r = 0.5 * (b11 * (b22 * b33 - b23 * b23) - b12 * (b12 * b33 - b23 * b13) +
                       b13 * (b12 * b23 - b22 * b13));
        if (r < -1.0) r = -1.0;
        if (r > 1.0) r = 1.0;
        const f64 phi = keel::acos(r) / 3.0;
        l1 = q + 2.0 * p * keel::cos(phi);
        l3 = q + 2.0 * p * keel::cos(phi + kTau / 3.0);
        l2 = 3.0 * q - l1 - l3;
    }
    if (keel::abs(l1) > keel::abs(l2)) keel::swap(l1, l2);
    if (keel::abs(l2) > keel::abs(l3)) keel::swap(l2, l3);
    if (keel::abs(l1) > keel::abs(l2)) keel::swap(l1, l2);
}

} // namespace

ImageStatus fibermetric_2d(ArrayRef<f64> img_col_major, usize height, usize width,
                           ArrayRef<f64> thickness, f64 sensitivity, bool bright,
                           MutArrayRef<f64> out_col_major, Allocator* alloc) {
    if (height < 2u || width < 2u || thickness.empty() || !keel::is_finite(sensitivity) ||
        !(sensitivity > 0.0)) {
        return ImageStatus::kInvalidArgument;
    }
    const usize n = height * width;
    if (img_col_major.size() < n || out_col_major.size() < n) {
        return ImageStatus::kInvalidArgument;
    }
    for (usize i = 0; i < thickness.size(); ++i) {
        if (!keel::is_finite(thickness[i]) || !(thickness[i] > 0.0)) {
            return ImageStatus::kInvalidArgument;
        }
    }

    for (usize i = 0; i < n; ++i) out_col_major[i] = 0.0;

    ArenaTemp scratch = get_scratch(alloc);
    Array<f64> smoothed(n, 0.0, scratch.allocator());
    const f64 beta_sq2 = 2.0 * kFrangiBeta * kFrangiBeta;
    const f64 c_sq2 = 2.0 * sensitivity * sensitivity;

    for (usize ti = 0; ti < thickness.size(); ++ti) {
        const f64 sigma = thickness[ti] / 6.0;
        const f64 sigma2 = sigma * sigma;
        gauss_2d_sep(img_col_major, height, width, sigma, smoothed.as_mut_ref(),
                     scratch.allocator());

        auto at = [&](usize r, usize c) noexcept -> f64 { return smoothed[c * height + r]; };
        for (usize cc = 0; cc < width; ++cc) {
            const usize cm = clamp_idx(as<i64>(cc) - 1, width);
            const usize cp = clamp_idx(as<i64>(cc) + 1, width);
            for (usize r = 0; r < height; ++r) {
                const usize rm = clamp_idx(as<i64>(r) - 1, height);
                const usize rp = clamp_idx(as<i64>(r) + 1, height);
                const f64 v = at(r, cc);
                const f64 ixx = sigma2 * (at(r, cp) - 2.0 * v + at(r, cm));
                const f64 iyy = sigma2 * (at(rp, cc) - 2.0 * v + at(rm, cc));
                const f64 ixy = sigma2 * 0.25 * (at(rp, cp) - at(rp, cm) - at(rm, cp) + at(rm, cm));

                const f64 mean = 0.5 * (ixx + iyy);
                const f64 diff = 0.5 * (ixx - iyy);
                const f64 disc = keel::sqrt(diff * diff + ixy * ixy);
                f64 l1 = mean - disc;
                f64 l2 = mean + disc;
                if (keel::abs(l1) > keel::abs(l2)) keel::swap(l1, l2);

                if (bright ? (l2 > 0.0) : (l2 < 0.0)) continue;
                if (l2 == 0.0) continue;

                const f64 rb_sq = (l1 / l2) * (l1 / l2);
                const f64 s_sq = l1 * l1 + l2 * l2;
                const f64 resp =
                    keel::exp(-rb_sq / beta_sq2) * (1.0 - keel::exp(-s_sq / c_sq2));
                const usize idx = cc * height + r;
                if (resp > out_col_major[idx]) out_col_major[idx] = resp;
            }
        }
    }
    return ImageStatus::kOk;
}

ImageStatus fibermetric_3d(ArrayRef<f64> vol_col_major, usize height, usize width, usize depth,
                           ArrayRef<f64> thickness, f64 sensitivity, bool bright,
                           MutArrayRef<f64> out_col_major, Allocator* alloc) {
    if (height < 2u || width < 2u || depth < 2u || thickness.empty() ||
        !keel::is_finite(sensitivity) || !(sensitivity > 0.0)) {
        return ImageStatus::kInvalidArgument;
    }
    const usize hw = height * width;
    const usize n = hw * depth;
    if (vol_col_major.size() < n || out_col_major.size() < n) {
        return ImageStatus::kInvalidArgument;
    }
    for (usize i = 0; i < thickness.size(); ++i) {
        if (!keel::is_finite(thickness[i]) || !(thickness[i] > 0.0)) {
            return ImageStatus::kInvalidArgument;
        }
    }

    for (usize i = 0; i < n; ++i) out_col_major[i] = 0.0;

    ArenaTemp scratch = get_scratch(alloc);
    Array<f64> smoothed(n, 0.0, scratch.allocator());
    const f64 alpha_sq2 = 2.0 * kFrangiAlpha * kFrangiAlpha;
    const f64 beta_sq2 = 2.0 * kFrangiBeta * kFrangiBeta;
    const f64 c_sq2 = 2.0 * sensitivity * sensitivity;

    for (usize ti = 0; ti < thickness.size(); ++ti) {
        const f64 sigma = thickness[ti] / 6.0;
        const f64 sigma2 = sigma * sigma;
        gauss_3d_sep(vol_col_major, height, width, depth, sigma, smoothed.as_mut_ref(),
                     scratch.allocator());

        auto at = [&](usize r, usize c, usize z) noexcept -> f64 {
            return smoothed[z * hw + c * height + r];
        };
        for (usize z = 0; z < depth; ++z) {
            const usize zm = clamp_idx(as<i64>(z) - 1, depth);
            const usize zp = clamp_idx(as<i64>(z) + 1, depth);
            for (usize cc = 0; cc < width; ++cc) {
                const usize cm = clamp_idx(as<i64>(cc) - 1, width);
                const usize cp = clamp_idx(as<i64>(cc) + 1, width);
                for (usize r = 0; r < height; ++r) {
                    const usize rm = clamp_idx(as<i64>(r) - 1, height);
                    const usize rp = clamp_idx(as<i64>(r) + 1, height);
                    const f64 v = at(r, cc, z);
                    const f64 ixx = sigma2 * (at(r, cp, z) - 2.0 * v + at(r, cm, z));
                    const f64 iyy = sigma2 * (at(rp, cc, z) - 2.0 * v + at(rm, cc, z));
                    const f64 izz = sigma2 * (at(r, cc, zp) - 2.0 * v + at(r, cc, zm));
                    const f64 ixy = sigma2 * 0.25 *
                                    (at(rp, cp, z) - at(rp, cm, z) - at(rm, cp, z) + at(rm, cm, z));
                    const f64 ixz = sigma2 * 0.25 *
                                    (at(r, cp, zp) - at(r, cm, zp) - at(r, cp, zm) + at(r, cm, zm));
                    const f64 iyz = sigma2 * 0.25 *
                                    (at(rp, cc, zp) - at(rm, cc, zp) - at(rp, cc, zm) + at(rm, cc, zm));

                    f64 l1 = 0.0, l2 = 0.0, l3 = 0.0;
                    eigvals_3x3_sym(ixx, iyy, izz, ixy, ixz, iyz, l1, l2, l3);
                    if (bright) {
                        if (l2 > 0.0 || l3 > 0.0) continue;
                    } else {
                        if (l2 < 0.0 || l3 < 0.0) continue;
                    }
                    if (l2 == 0.0 || l3 == 0.0) continue;
                    const f64 ra_sq = (l2 / l3) * (l2 / l3);
                    const f64 prod23 = keel::abs(l2 * l3);
                    if (prod23 == 0.0) continue;
                    const f64 rb_sq = (l1 * l1) / prod23;
                    const f64 s_sq = l1 * l1 + l2 * l2 + l3 * l3;
                    const f64 resp = (1.0 - keel::exp(-ra_sq / alpha_sq2)) *
                                     keel::exp(-rb_sq / beta_sq2) *
                                     (1.0 - keel::exp(-s_sq / c_sq2));
                    const usize idx = z * hw + cc * height + r;
                    if (resp > out_col_major[idx]) out_col_major[idx] = resp;
                }
            }
        }
    }
    return ImageStatus::kOk;
}

} // namespace keel::image
