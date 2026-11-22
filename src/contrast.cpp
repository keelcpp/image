#include "keel/image.h"

#include "keel/math.h"

namespace keel::image {

namespace {

inline i32 clamp_idx(i32 i, i32 n) noexcept {
    if (i < 0) return 0;
    if (i >= n) return n - 1;
    return i;
}

inline i32 sym_half_idx(i32 i, i32 n) noexcept {
    if (n <= 1) return 0;
    const i32 cycle = 2 * n;
    i32 m = ((i % cycle) + cycle) % cycle;
    if (m >= n) m = cycle - 1 - m;
    return m;
}

inline usize mirror_no_edge(usize i, usize n) noexcept {
    if (n <= 1) return 0;
    if (i < n) return i;
    const usize period = 2 * n - 2;
    const usize k = (period > 0) ? (i % period) : 0;
    return (k < n) ? k : (period - k);
}

void clip_histogram(f64* hist, i32 nbins, f64 clip_limit) noexcept {
    f64 total_excess = 0.0;
    for (i32 b = 0; b < nbins; ++b) {
        if (hist[b] > clip_limit) total_excess += hist[b] - clip_limit;
    }

    const f64 avg_incr = floor(total_excess / as<f64>(nbins));
    const f64 upper_limit = clip_limit - avg_incr;
    for (i32 k = 0; k < nbins; ++k) {
        if (hist[k] > clip_limit) {
            hist[k] = clip_limit;
        } else if (hist[k] > upper_limit) {
            total_excess -= (clip_limit - hist[k]);
            hist[k] = clip_limit;
        } else {
            total_excess -= avg_incr;
            hist[k] += avg_incr;
        }
    }

    i32 k = 0;
    while (total_excess > 0.0) {
        i32 step = as<i32>(floor(as<f64>(nbins) / total_excess));
        if (step < 1) step = 1;
        for (i32 m = k; m < nbins; m += step) {
            if (hist[m] < clip_limit) {
                hist[m] += 1.0;
                total_excess -= 1.0;
                if (total_excess <= 0.0) break;
            }
        }
        if (++k >= nbins) k = 0;
    }
}

f64 map_probability(f64 p, ClaheDistribution dist, f64 alpha, f64 out_min, f64 out_max) noexcept {
    const f64 span = out_max - out_min;
    constexpr f64 kEps = Limits<f64>::kEpsilon;
    switch (dist) {
        case ClaheDistribution::kRayleigh: {
            const f64 hconst = 2.0 * alpha * alpha;
            const f64 vmax = 1.0 - exp(-1.0 / hconst);
            f64 val = vmax * p;
            if (val >= 1.0) val = 1.0 - kEps;
            const f64 temp = sqrt(-hconst * log(1.0 - val));
            return min(out_min + temp * span, out_max);
        }
        case ClaheDistribution::kExponential: {
            const f64 vmax = 1.0 - exp(-alpha);
            f64 val = vmax * p;
            if (val >= 1.0) val = 1.0 - kEps;
            const f64 temp = -(1.0 / alpha) * log(1.0 - val);
            return min(out_min + temp * span, out_max);
        }
        case ClaheDistribution::kUniform:
        default:
            return min(out_min + p * span, out_max);
    }
}

inline i32 level_of_unit(f64 u, i32 bins) noexcept {
    i32 b = as<i32>(round(u * as<f64>(bins - 1)));
    if (b < 0) b = 0;
    if (b >= bins) b = bins - 1;
    return b;
}

} // namespace

ImageStatus gaussian_filter_2d(ArrayRef<f64> in_col_major, usize height, usize width, f64 sigma,
                               i32 filter_size, bool symmetric_pad, MutArrayRef<f64> out_col_major,
                               Allocator* alloc) {
    const usize n = height * width;
    if (in_col_major.size() != n || out_col_major.size() != n) {
        return ImageStatus::kInvalidArgument;
    }
    if (height == 0 || width == 0) return ImageStatus::kOk;
    if (filter_size <= 0) {
        filter_size = 2 * as<i32>(ceil(2.0 * sigma)) + 1;
        if (filter_size < 3) filter_size = 3;
    }
    if ((filter_size % 2) == 0) filter_size += 1;
    if (!(sigma > 0.0)) {
        for (usize i = 0; i < n; ++i) out_col_major[i] = in_col_major[i];
        return ImageStatus::kOk;
    }

    const i32 half = (filter_size - 1) / 2;
    const f64 cx = as<f64>(filter_size - 1) / 2.0;
    const f64 inv2s2 = 1.0 / (2.0 * sigma * sigma);
    Array<f64> g(as<usize>(filter_size), 0.0, alloc);
    f64 gsum = 0.0;
    for (i32 i = 0; i < filter_size; ++i) {
        const f64 d = as<f64>(i) - cx;
        g[as<usize>(i)] = exp(-d * d * inv2s2);
        gsum += g[as<usize>(i)];
    }
    if (gsum > 0.0) {
        for (i32 i = 0; i < filter_size; ++i) g[as<usize>(i)] /= gsum;
    }

    const i32 ih = as<i32>(height);
    const i32 iw = as<i32>(width);
    Array<f64> tmp(n, 0.0, alloc);

    // Pass 1: horizontal (along cols)
    for (i32 c = 0; c < iw; ++c) {
        const usize dst_col = as<usize>(c) * height;
        for (i32 r = 0; r < ih; ++r) {
            f64 acc = 0.0;
            for (i32 k = 0; k < filter_size; ++k) {
                const i32 sc = symmetric_pad ? sym_half_idx(c + k - half, iw)
                                             : clamp_idx(c + k - half, iw);
                acc += g[as<usize>(k)] * in_col_major[as<usize>(sc) * height + as<usize>(r)];
            }
            tmp[dst_col + as<usize>(r)] = acc;
        }
    }

    // Pass 2: vertical (along rows)
    for (i32 c = 0; c < iw; ++c) {
        const usize col_off = as<usize>(c) * height;
        for (i32 r = 0; r < ih; ++r) {
            f64 acc = 0.0;
            for (i32 k = 0; k < filter_size; ++k) {
                const i32 sr = symmetric_pad ? sym_half_idx(r + k - half, ih)
                                             : clamp_idx(r + k - half, ih);
                acc += g[as<usize>(k)] * tmp[col_off + as<usize>(sr)];
            }
            out_col_major[col_off + as<usize>(r)] = acc;
        }
    }
    return ImageStatus::kOk;
}

ImageStatus stretchlim_unit(ArrayRef<f64> unit_planar, usize plane_len, usize channels, i32 nbins,
                            f64 low_tol, f64 high_tol, MutArrayRef<f64> out_lo_hi,
                            Allocator* alloc) {
    if (plane_len == 0 || channels == 0 || nbins < 2 ||
        unit_planar.size() != plane_len * channels || out_lo_hi.size() != 2 * channels) {
        return ImageStatus::kInvalidArgument;
    }
    if (low_tol < 0.0) low_tol = 0.01;
    if (high_tol > 1.0 || high_tol <= low_tol) high_tol = 0.99;

    Array<usize> hist(as<usize>(nbins), usize{0}, alloc);
    const f64 nf = as<f64>(plane_len);

    for (usize ch = 0; ch < channels; ++ch) {
        for (i32 k = 0; k < nbins; ++k) hist[as<usize>(k)] = 0;
        const usize off = ch * plane_len;
        for (usize i = 0; i < plane_len; ++i) {
            const i32 bin = level_of_unit(unit_planar[off + i], nbins);
            hist[as<usize>(bin)]++;
        }
        usize cum = 0;
        i32 low = nbins - 1;
        i32 high = nbins - 1;
        bool found_low = false;
        bool found_high = false;
        for (i32 k = 0; k < nbins; ++k) {
            cum += hist[as<usize>(k)];
            if (!found_low && as<f64>(cum) > low_tol * nf) {
                low = k;
                found_low = true;
            }
            if (!found_high && as<f64>(cum) >= high_tol * nf) {
                high = k;
                found_high = true;
            }
            if (found_low && found_high) break;
        }
        if (high <= low) high = min(nbins - 1, low + 1);
        out_lo_hi[ch * 2 + 0] = as<f64>(low) / as<f64>(nbins - 1);
        out_lo_hi[ch * 2 + 1] = as<f64>(high) / as<f64>(nbins - 1);
    }
    return ImageStatus::kOk;
}

ImageStatus imadjust_unit(ArrayRef<f64> unit_in, f64 low_in, f64 high_in, f64 low_out,
                          f64 high_out, f64 gamma, MutArrayRef<f64> unit_out) noexcept {
    if (unit_in.size() != unit_out.size()) return ImageStatus::kInvalidArgument;
    if (is_nan(low_out)) low_out = 0.0;
    if (is_nan(high_out)) high_out = 1.0;
    if (is_nan(gamma) || gamma <= 0.0) gamma = 1.0;

    const f64 range = high_in - low_in;
    const f64 inv_range = (range != 0.0) ? (1.0 / range) : 0.0;
    const f64 out_span = high_out - low_out;
    for (usize i = 0; i < unit_in.size(); ++i) {
        const f64 t = clamp((unit_in[i] - low_in) * inv_range, 0.0, 1.0);
        unit_out[i] = pow(t, gamma) * out_span + low_out;
    }
    return ImageStatus::kOk;
}

ImageStatus histeq_unit(ArrayRef<f64> unit_in, i32 nbins, MutArrayRef<f64> unit_out,
                        Allocator* alloc) {
    if (unit_in.size() != unit_out.size()) return ImageStatus::kInvalidArgument;
    if (nbins <= 0) nbins = 64;
    if (nbins < 2) nbins = 2;
    const usize n = unit_in.size();
    if (n == 0) return ImageStatus::kOk;

    Array<f64> counts(as<usize>(nbins), 0.0, alloc);
    f64 total = 0.0;
    for (usize i = 0; i < n; ++i) {
        const f64 u = unit_in[i];
        if (is_nan(u)) continue;
        counts[as<usize>(level_of_unit(u, nbins))] += 1.0;
        total += 1.0;
    }
    if (total <= 0.0) {
        for (usize i = 0; i < n; ++i) unit_out[i] = unit_in[i];
        return ImageStatus::kOk;
    }

    Array<f64> cdf(as<usize>(nbins), 0.0, alloc);
    f64 acc = 0.0;
    for (i32 i = 0; i < nbins; ++i) {
        acc += counts[as<usize>(i)];
        cdf[as<usize>(i)] = acc / total;
    }
    for (usize i = 0; i < n; ++i) {
        unit_out[i] = cdf[as<usize>(level_of_unit(unit_in[i], nbins))];
    }
    return ImageStatus::kOk;
}

ImageStatus clahe_2d(ArrayRef<f64> img_col_major, usize height, usize width, i32 num_tiles_r,
                     i32 num_tiles_c, f64 clip_limit, i32 nbins, ClaheDistribution dist, f64 alpha,
                     f64 class_lo, f64 class_hi, bool range_original,
                     MutArrayRef<f64> out_col_major, Allocator* alloc) {
    if (height == 0 || width == 0 || img_col_major.size() != height * width ||
        out_col_major.size() != height * width) {
        return ImageStatus::kInvalidArgument;
    }
    if (num_tiles_r < 2 || num_tiles_c < 2 || nbins < 2 ||
        !(clip_limit >= 0.0 && clip_limit <= 1.0)) {
        return ImageStatus::kInvalidArgument;
    }

    f64 in_min = class_lo;
    f64 in_max = class_hi;
    f64 out_min = class_lo;
    f64 out_max = class_hi;
    if (range_original) {
        f64 mn = Limits<f64>::infinity();
        f64 mx = -Limits<f64>::infinity();
        for (usize i = 0; i < height * width; ++i) {
            const f64 v = img_col_major[i];
            if (v < mn) mn = v;
            if (v > mx) mx = v;
        }
        if (!is_finite(mn) || !is_finite(mx)) {
            mn = class_lo;
            mx = class_hi;
        }
        in_min = out_min = mn;
        in_max = out_max = mx;
    }
    if (!(in_max > in_min)) {
        in_min = class_lo;
        in_max = class_hi;
    }
    if (!(in_max > in_min)) {
        in_min = 0.0;
        in_max = 1.0;
    }

    const usize unit_r = 2 * as<usize>(num_tiles_r);
    const usize unit_c = 2 * as<usize>(num_tiles_c);
    const usize pad_b = (unit_r - (height % unit_r)) % unit_r;
    const usize pad_r = (unit_c - (width % unit_c)) % unit_c;
    const usize hp = height + pad_b;
    const usize wp = width + pad_r;
    const usize tile_h = hp / as<usize>(num_tiles_r);
    const usize tile_w = wp / as<usize>(num_tiles_c);

    Array<f64> img(hp * wp, 0.0, alloc);
    for (usize c = 0; c < wp; ++c) {
        const usize sc = mirror_no_edge(c, width);
        for (usize r = 0; r < hp; ++r) {
            const usize sr = mirror_no_edge(r, height);
            img[c * hp + r] = img_col_major[sc * height + sr];
        }
    }

    const f64 inv_span = as<f64>(nbins) / (in_max - in_min);
    auto bin_of = [&](f64 v) noexcept -> i32 {
        i32 b = as<i32>((v - in_min) * inv_span);
        if (b < 0) b = 0;
        if (b >= nbins) b = nbins - 1;
        return b;
    };

    const usize n_tiles = as<usize>(num_tiles_r) * as<usize>(num_tiles_c);
    const f64 num_pix = as<f64>(tile_h * tile_w);
    const f64 min_clip = ceil(num_pix / as<f64>(nbins));
    const f64 clip_count = min_clip + round(clip_limit * (num_pix - min_clip));

    Array<f64> luts(n_tiles * as<usize>(nbins), 0.0, alloc);
    Array<f64> hist(as<usize>(nbins), 0.0, alloc);

    for (i32 tr = 0; tr < num_tiles_r; ++tr) {
        for (i32 tc = 0; tc < num_tiles_c; ++tc) {
            for (i32 b = 0; b < nbins; ++b) hist[as<usize>(b)] = 0.0;
            const usize r0 = as<usize>(tr) * tile_h;
            const usize c0 = as<usize>(tc) * tile_w;
            for (usize cc = 0; cc < tile_w; ++cc) {
                for (usize rr = 0; rr < tile_h; ++rr) {
                    hist[as<usize>(bin_of(img[(c0 + cc) * hp + (r0 + rr)]))] += 1.0;
                }
            }
            clip_histogram(hist.data(), nbins, clip_count);
            f64* lut =
                luts.data() + (as<usize>(tr) * as<usize>(num_tiles_c) + as<usize>(tc)) * as<usize>(nbins);
            f64 cum = 0.0;
            for (i32 b = 0; b < nbins; ++b) {
                cum += hist[as<usize>(b)];
                lut[b] = map_probability(cum / num_pix, dist, alpha, out_min, out_max);
            }
        }
    }

    auto lut_at = [&](i32 tr, i32 tc) noexcept -> const f64* {
        return luts.data() +
               (as<usize>(tr) * as<usize>(num_tiles_c) + as<usize>(tc)) * as<usize>(nbins);
    };

    Array<f64> out_pad(hp * wp, 0.0, alloc);
    usize reg_row0 = 0;
    for (i32 kr = 0; kr <= num_tiles_r; ++kr) {
        usize reg_h = 0;
        i32 tr_u = 0;
        i32 tr_l = 0;
        if (kr == 0) {
            reg_h = tile_h / 2;
            tr_u = 0;
            tr_l = 0;
        } else if (kr == num_tiles_r) {
            reg_h = tile_h / 2;
            tr_u = num_tiles_r - 1;
            tr_l = num_tiles_r - 1;
        } else {
            reg_h = tile_h;
            tr_u = kr - 1;
            tr_l = kr;
        }

        usize reg_col0 = 0;
        for (i32 kc = 0; kc <= num_tiles_c; ++kc) {
            usize reg_w = 0;
            i32 tc_l = 0;
            i32 tc_r = 0;
            if (kc == 0) {
                reg_w = tile_w / 2;
                tc_l = 0;
                tc_r = 0;
            } else if (kc == num_tiles_c) {
                reg_w = tile_w / 2;
                tc_l = num_tiles_c - 1;
                tc_r = num_tiles_c - 1;
            } else {
                reg_w = tile_w;
                tc_l = kc - 1;
                tc_r = kc;
            }

            const f64* lut_ul = lut_at(tr_u, tc_l);
            const f64* lut_ur = lut_at(tr_u, tc_r);
            const f64* lut_bl = lut_at(tr_l, tc_l);
            const f64* lut_br = lut_at(tr_l, tc_r);
            const f64 norm_factor = as<f64>(reg_h) * as<f64>(reg_w);

            for (usize rr = 0; rr < reg_h; ++rr) {
                const usize r = reg_row0 + rr;
                const f64 row_w = as<f64>(rr);
                const f64 row_rev = as<f64>(reg_h - rr);
                for (usize cc = 0; cc < reg_w; ++cc) {
                    const usize c = reg_col0 + cc;
                    const f64 col_w = as<f64>(cc);
                    const f64 col_rev = as<f64>(reg_w - cc);
                    const i32 bin = bin_of(img[c * hp + r]);
                    out_pad[c * hp + r] =
                        (row_rev * (col_rev * lut_ul[bin] + col_w * lut_ur[bin]) +
                         row_w * (col_rev * lut_bl[bin] + col_w * lut_br[bin])) /
                        norm_factor;
                }
            }
            reg_col0 += reg_w;
        }
        reg_row0 += reg_h;
    }

    const f64 lo = min(out_min, out_max);
    const f64 hi = max(out_min, out_max);
    for (usize c = 0; c < width; ++c) {
        for (usize r = 0; r < height; ++r) {
            out_col_major[c * height + r] = clamp(out_pad[c * hp + r], lo, hi);
        }
    }
    return ImageStatus::kOk;
}

ImageStatus multithresh_otsu(ArrayRef<f64> data, i32 num_levels, bool integer_scale,
                             MutArrayRef<f64> out_thresh, f64& out_effectiveness,
                             Allocator* alloc) {
    if (num_levels < 1 || num_levels > 5 || out_thresh.size() != as<usize>(num_levels)) {
        return ImageStatus::kInvalidArgument;
    }

    constexpr i32 kBins = 256;
    const usize m = data.size();
    f64 min_a = 0.0;
    f64 max_a = 0.0;
    bool any_finite = false;
    for (usize i = 0; i < m; ++i) {
        const f64 x = data[i];
        if (!is_finite(x)) continue;
        if (!any_finite) {
            min_a = max_a = x;
            any_finite = true;
        } else {
            if (x < min_a) min_a = x;
            if (x > max_a) max_a = x;
        }
    }

    auto to_bin = [&](f64 x) noexcept -> i32 {
        if (is_nan(x)) return -1;
        if (max_a == min_a) return 0;
        if (is_inf(x)) return (x > 0.0) ? 255 : 0;
        i64 b = 0;
        if (integer_scale) {
            const f32 fs = as<f32>(x - min_a) / as<f32>(max_a - min_a);
            b = as<i64>(round(as<f64>(fs * 255.0f)));
        } else {
            const f64 s = (x - min_a) / (max_a - min_a);
            b = as<i64>(round(s * 255.0));
        }
        return clamp(as<i32>(b), i32{0}, i32{255});
    };

    f64 counts[kBins]{};
    f64 total = 0.0;
    for (usize i = 0; i < m; ++i) {
        const i32 b = to_bin(data[i]);
        if (b >= 0) {
            counts[b] += 1.0;
            total += 1.0;
        }
    }

    if (total <= 0.0 || max_a == min_a) {
        for (i32 k = 0; k < num_levels; ++k) out_thresh[as<usize>(k)] = min_a;
        out_effectiveness = 0.0;
        return ImageStatus::kOk;
    }

    f64 omega[kBins]{};
    f64 mu[kBins]{};
    f64 cum = 0.0;
    f64 cumm = 0.0;
    for (i32 j = 0; j < kBins; ++j) {
        const f64 p = counts[j] / total;
        cum += p;
        cumm += p * as<f64>(j + 1);
        omega[j] = cum;
        mu[j] = cumm;
    }
    const f64 mu_t = mu[kBins - 1];

    f64 thresh_bins[5]{};
    f64 best_var = 0.0;

    if (num_levels == 1) {
        f64 best = -Limits<f64>::infinity();
        f64 tie_sum = 0.0;
        usize tie_cnt = 0;
        for (i32 t = 0; t < kBins; ++t) {
            const f64 o = omega[t];
            if (o <= 0.0 || o >= 1.0) continue;
            const f64 num = mu_t * o - mu[t];
            const f64 s = num * num / (o * (1.0 - o));
            if (!is_finite(s)) continue;
            if (s > best) {
                best = s;
                tie_sum = as<f64>(t);
                tie_cnt = 1;
            } else if (s == best) {
                tie_sum += as<f64>(t);
                tie_cnt += 1;
            }
        }
        thresh_bins[0] = (tie_cnt > 0) ? (tie_sum / as<f64>(tie_cnt)) : 0.0;
        best_var = best;
    } else if (num_levels == 2) {
        f64 best = -Limits<f64>::infinity();
        f64 sr = 0.0;
        f64 sc = 0.0;
        usize tie_cnt = 0;
        for (i32 r = 0; r < kBins; ++r) {
            const f64 o0 = omega[r];
            if (o0 <= 0.0) continue;
            const f64 mu0t = mu_t - mu[r] / o0;
            for (i32 c = r + 1; c < kBins; ++c) {
                const f64 o1 = omega[c] - omega[r];
                if (o1 <= 0.0) continue;
                const f64 mu1t = mu_t - (mu[c] - mu[r]) / o1;
                const f64 o2 = 1.0 - (o0 + o1);
                if (o2 <= 0.0) continue;
                const f64 term1 = o0 * mu0t * mu0t;
                const f64 term2 = o1 * mu1t * mu1t;
                const f64 q = o0 * mu0t + o1 * mu1t;
                const f64 term3 = q * q / o2;
                const f64 s = term1 + term2 + term3;
                if (!is_finite(s)) continue;
                if (s > best) {
                    best = s;
                    sr = as<f64>(r);
                    sc = as<f64>(c);
                    tie_cnt = 1;
                } else if (s == best) {
                    sr += as<f64>(r);
                    sc += as<f64>(c);
                    tie_cnt += 1;
                }
            }
        }
        const f64 den = (tie_cnt > 0) ? as<f64>(tie_cnt) : 1.0;
        thresh_bins[0] = sr / den;
        thresh_bins[1] = sc / den;
        best_var = best;
    } else {
        f64 o_pre[kBins + 1]{};
        f64 m_pre[kBins + 1]{};
        for (i32 j = 0; j < kBins; ++j) {
            o_pre[j + 1] = omega[j];
            m_pre[j + 1] = mu[j];
        }
        auto seg_score = [&](i32 lo, i32 hi) noexcept -> f64 {
            const f64 w = o_pre[hi + 1] - o_pre[lo];
            if (w <= 0.0) return 0.0;
            const f64 s = m_pre[hi + 1] - m_pre[lo];
            return s * s / w;
        };
        const i32 K = num_levels + 1;
        const f64 neg = -Limits<f64>::infinity();
        Array<f64> dp(as<usize>(K + 1) * kBins, neg, alloc);
        Array<i32> back(as<usize>(K + 1) * kBins, i32{-1}, alloc);
        for (i32 t = 0; t < kBins; ++t) dp[as<usize>(kBins + t)] = seg_score(0, t);
        for (i32 k = 2; k <= K; ++k) {
            for (i32 t = k - 1; t < kBins; ++t) {
                f64 best_v = neg;
                i32 best_p = -1;
                for (i32 tp = k - 2; tp < t; ++tp) {
                    const f64 prev = dp[as<usize>((k - 1) * kBins + tp)];
                    if (prev == neg) continue;
                    const f64 v = prev + seg_score(tp + 1, t);
                    if (v > best_v) {
                        best_v = v;
                        best_p = tp;
                    }
                }
                dp[as<usize>(k * kBins + t)] = best_v;
                back[as<usize>(k * kBins + t)] = best_p;
            }
        }
        i32 bounds[6]{};
        bounds[K - 1] = kBins - 1;
        for (i32 k = K; k >= 2; --k) {
            bounds[k - 2] = back[as<usize>(k * kBins + bounds[k - 1])];
        }
        for (i32 k = 0; k < num_levels; ++k) thresh_bins[k] = as<f64>(bounds[k]);
        best_var = dp[as<usize>(K * kBins + (kBins - 1))] - mu_t * mu_t;
    }

    for (i32 k = 0; k < num_levels; ++k) {
        f64 v = min_a + thresh_bins[k] / 255.0 * (max_a - min_a);
        if (integer_scale) v = round(v);
        out_thresh[as<usize>(k)] = v;
    }

    f64 denom = 0.0;
    for (i32 j = 0; j < kBins; ++j) {
        const f64 p = counts[j] / total;
        const f64 d = as<f64>(j + 1) - mu_t;
        denom += p * d * d;
    }
    out_effectiveness = (denom > 0.0 && is_finite(best_var)) ? (best_var / denom) : 0.0;
    return ImageStatus::kOk;
}

ImageStatus imhistmatch_unit(ArrayRef<f64> in_unit, ArrayRef<f64> ref_unit, i32 npts, i32 nbins,
                             MutArrayRef<f64> out_unit, MutArrayRef<f64> out_hgram,
                             Allocator* alloc) {
    if (nbins <= 0) nbins = 64;
    if (nbins < 2) nbins = 2;
    if (npts < 2 || in_unit.size() != out_unit.size()) return ImageStatus::kInvalidArgument;
    if (!out_hgram.empty() && out_hgram.size() != as<usize>(nbins)) {
        return ImageStatus::kInvalidArgument;
    }

    const usize ni = in_unit.size();
    const usize nr = ref_unit.size();
    if (ni == 0 || nr == 0) {
        if (!out_hgram.empty()) {
            for (i32 b = 0; b < nbins; ++b) out_hgram[as<usize>(b)] = 0.0;
        }
        return ImageStatus::kOk;
    }

    Array<f64> hgram(as<usize>(nbins), 0.0, alloc);
    for (usize i = 0; i < nr; ++i) {
        hgram[as<usize>(level_of_unit(ref_unit[i], nbins))] += 1.0;
    }
    if (!out_hgram.empty()) {
        for (i32 b = 0; b < nbins; ++b) out_hgram[as<usize>(b)] = hgram[as<usize>(b)];
    }

    f64 sum_h = 0.0;
    for (i32 b = 0; b < nbins; ++b) sum_h += hgram[as<usize>(b)];
    const f64 scale = (sum_h > 0.0) ? (as<f64>(ni) / sum_h) : 0.0;
    Array<f64> cumd(as<usize>(nbins), 0.0, alloc);
    {
        f64 acc = 0.0;
        for (i32 b = 0; b < nbins; ++b) {
            acc += hgram[as<usize>(b)] * scale;
            cumd[as<usize>(b)] = acc;
        }
    }

    Array<f64> nn(as<usize>(npts), 0.0, alloc);
    for (usize i = 0; i < ni; ++i) {
        nn[as<usize>(level_of_unit(in_unit[i], npts))] += 1.0;
    }
    Array<f64> cum(as<usize>(npts), 0.0, alloc);
    {
        f64 acc = 0.0;
        for (i32 j = 0; j < npts; ++j) {
            acc += nn[as<usize>(j)];
            cum[as<usize>(j)] = acc;
        }
    }

    const f64 clamp_thr = -as<f64>(ni) * sqrt(Limits<f64>::kEpsilon);
    const f64 denom = (nbins > 1) ? as<f64>(nbins - 1) : 1.0;
    Array<f64> T(as<usize>(npts), 0.0, alloc);
    for (i32 j = 0; j < npts; ++j) {
        const f64 tol = (j == 0 || j == npts - 1) ? 0.0 : (nn[as<usize>(j)] * 0.5);
        const f64 base = tol - cum[as<usize>(j)];
        f64 best = 0.0;
        i32 bi = 0;
        bool have = false;
        for (i32 i = 0; i < nbins; ++i) {
            f64 e = cumd[as<usize>(i)] + base;
            if (e < clamp_thr) e = as<f64>(ni);
            if (!have || e < best) {
                best = e;
                bi = i;
                have = true;
            }
        }
        T[as<usize>(j)] = as<f64>(bi) / denom;
    }

    for (usize i = 0; i < ni; ++i) {
        out_unit[i] = T[as<usize>(level_of_unit(in_unit[i], npts))];
    }
    return ImageStatus::kOk;
}

ImageStatus adaptthresh_2d(ArrayRef<f64> unit_col_major, usize height, usize width,
                           f64 sensitivity, i32 neighborhood, bool gaussian,
                           MutArrayRef<f64> out_col_major, Allocator* alloc) {
    const usize n = height * width;
    if (unit_col_major.size() != n || out_col_major.size() != n) {
        return ImageStatus::kInvalidArgument;
    }
    if (!(sensitivity >= 0.0 && sensitivity <= 1.0)) return ImageStatus::kInvalidArgument;
    if (height == 0 || width == 0) return ImageStatus::kOk;

    if (neighborhood <= 0) {
        const i32 dim = as<i32>(min(height, width));
        neighborhood = 2 * (dim / 16) + 1;
    }
    if ((neighborhood & 1) == 0) neighborhood += 1;

    Array<f64> local_stat(n, 0.0, alloc);
    if (gaussian) {
        const f64 sigma = as<f64>(neighborhood) / 6.0;
        (void)gaussian_filter_2d(unit_col_major, height, width, sigma, neighborhood, false,
                                 local_stat.as_mut_ref(), alloc);
    } else {
        (void)box_filter_2d_replicate(unit_col_major, height, width, neighborhood,
                                      local_stat.as_mut_ref(), alloc);
    }

    const f64 scale = 1.6 - sensitivity;
    for (usize i = 0; i < n; ++i) {
        out_col_major[i] = clamp(local_stat[i] * scale, 0.0, 1.0);
    }
    return ImageStatus::kOk;
}

ImageStatus flatfield_2d(ArrayRef<f64> unit_planar_col_major, usize height, usize width,
                         usize pages, f64 sigma, ArrayRef<u8> mask_col_major,
                         MutArrayRef<f64> out_planar_col_major, Allocator* alloc) {
    const usize plane = height * width;
    const usize total = plane * pages;
    if (unit_planar_col_major.size() != total || out_planar_col_major.size() != total) {
        return ImageStatus::kInvalidArgument;
    }
    if (!mask_col_major.empty() && mask_col_major.size() != plane) {
        return ImageStatus::kInvalidArgument;
    }
    if (plane == 0 || pages == 0) return ImageStatus::kOk;

    i32 filter_size = 2 * as<i32>(ceil(2.0 * sigma)) + 1;
    if (filter_size < 3) filter_size = 3;

    const bool have_mask = !mask_col_major.empty();
    Array<f64> blurred(plane, 0.0, alloc);
    constexpr f64 kEps = 1e-12;

    for (usize p = 0; p < pages; ++p) {
        const ArrayRef<f64> ap(unit_planar_col_major.data() + p * plane, plane);
        (void)gaussian_filter_2d(ap, height, width, sigma, filter_size, true, blurred.as_mut_ref(),
                                 alloc);

        f64 mean_a = 0.0;
        if (have_mask) {
            usize cnt = 0;
            f64 acc = 0.0;
            for (usize i = 0; i < plane; ++i) {
                if (mask_col_major[i] != 0) {
                    acc += ap[i];
                    ++cnt;
                }
            }
            mean_a = (cnt > 0) ? (acc / as<f64>(cnt)) : 0.0;
        } else {
            f64 acc = 0.0;
            for (usize i = 0; i < plane; ++i) acc += ap[i];
            mean_a = acc / as<f64>(plane);
        }

        f64* op = out_planar_col_major.data() + p * plane;
        for (usize i = 0; i < plane; ++i) {
            const f64 f = blurred[i];
            op[i] = (f > kEps) ? (ap[i] * mean_a / f) : ap[i];
        }
    }
    return ImageStatus::kOk;
}

ImageStatus wcodemat(ArrayRef<f64> x_col_major, usize height, usize width, usize pages, i32 nb,
                     WcodematMode mode, bool absol, MutArrayRef<f64> out_col_major) noexcept {
    const usize n = height * width * pages;
    if (nb < 1 || x_col_major.size() != n || out_col_major.size() != n) {
        return ImageStatus::kInvalidArgument;
    }
    if (n == 0) return ImageStatus::kOk;

    auto vget = [&](usize i) noexcept -> f64 {
        const f64 v = x_col_major[i];
        return absol ? abs(v) : v;
    };
    auto encode = [&](f64 v, f64 mn, f64 mx) noexcept -> f64 {
        const f64 span = mx - mn;
        if (span == 0.0) return 1.0;
        f64 y = floor((v - mn) / span * as<f64>(nb)) + 1.0;
        return clamp(y, 1.0, as<f64>(nb));
    };

    if (mode == WcodematMode::kMat) {
        f64 mn = Limits<f64>::infinity();
        f64 mx = -Limits<f64>::infinity();
        for (usize i = 0; i < n; ++i) {
            const f64 v = vget(i);
            if (v < mn) mn = v;
            if (v > mx) mx = v;
        }
        for (usize i = 0; i < n; ++i) out_col_major[i] = encode(vget(i), mn, mx);
    } else if (mode == WcodematMode::kRow) {
        for (usize r = 0; r < height; ++r) {
            f64 mn = Limits<f64>::infinity();
            f64 mx = -Limits<f64>::infinity();
            for (usize c = 0; c < width; ++c) {
                const f64 v = vget(c * height + r);
                if (v < mn) mn = v;
                if (v > mx) mx = v;
            }
            for (usize c = 0; c < width; ++c) {
                out_col_major[c * height + r] = encode(vget(c * height + r), mn, mx);
            }
        }
    } else {
        for (usize c = 0; c < width; ++c) {
            f64 mn = Limits<f64>::infinity();
            f64 mx = -Limits<f64>::infinity();
            for (usize r = 0; r < height; ++r) {
                const f64 v = vget(c * height + r);
                if (v < mn) mn = v;
                if (v > mx) mx = v;
            }
            for (usize r = 0; r < height; ++r) {
                out_col_major[c * height + r] = encode(vget(c * height + r), mn, mx);
            }
        }
    }
    return ImageStatus::kOk;
}

f64 shannon_entropy(ArrayRef<f64> hist_counts) noexcept {
    f64 total = 0.0;
    for (usize i = 0; i < hist_counts.size(); ++i) total += hist_counts[i];
    if (total <= 0.0) return 0.0;
    f64 h = 0.0;
    for (usize i = 0; i < hist_counts.size(); ++i) {
        if (hist_counts[i] <= 0.0) continue;
        const f64 p = hist_counts[i] / total;
        h -= p * (log(p) / kLn2);
    }
    return h;
}

} // namespace keel::image
