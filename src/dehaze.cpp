#include "keel/image.h"

#include "keel/math.h"

namespace keel::image {

namespace {

// Separable square-window erosion (IsErode=true) or dilation (IsErode=false)
// over in-bounds pixels matching unpack_se's half = k / 2 convention.
template <bool IsErode>
void morph_square_f32(ArrayRef<f32> src, usize h, usize w, i32 k, MutArrayRef<f32> dst,
                      Allocator* alloc) {
    if (k <= 1) {
        for (usize i = 0; i < h * w; ++i) dst[i] = src[i];
        return;
    }
    const i32 half = k / 2;
    const i32 right = k - 1 - half;
    const i32 ih = as<i32>(h);
    const i32 iw = as<i32>(w);

    Array<f32> tmp(h * w, 0.0f, alloc);
    // Pass 1: along rows (vertical sliding min/max).
    for (i32 c = 0; c < iw; ++c) {
        const usize col_off = as<usize>(c) * h;
        for (i32 r = 0; r < ih; ++r) {
            const i32 r0 = max(i32{0}, r - half);
            const i32 r1 = min(ih - 1, r + right);
            f32 best = src[col_off + as<usize>(r0)];
            for (i32 rr = r0 + 1; rr <= r1; ++rr) {
                const f32 v = src[col_off + as<usize>(rr)];
                if constexpr (IsErode) {
                    if (v < best) best = v;
                } else {
                    if (v > best) best = v;
                }
            }
            tmp[col_off + as<usize>(r)] = best;
        }
    }
    // Pass 2: along cols (horizontal sliding min/max).
    for (i32 c = 0; c < iw; ++c) {
        const i32 c0 = max(i32{0}, c - half);
        const i32 c1 = min(iw - 1, c + right);
        const usize dst_col = as<usize>(c) * h;
        for (i32 r = 0; r < ih; ++r) {
            f32 best = tmp[as<usize>(c0) * h + as<usize>(r)];
            for (i32 cc = c0 + 1; cc <= c1; ++cc) {
                const f32 v = tmp[as<usize>(cc) * h + as<usize>(r)];
                if constexpr (IsErode) {
                    if (v < best) best = v;
                } else {
                    if (v > best) best = v;
                }
            }
            dst[dst_col + as<usize>(r)] = best;
        }
    }
}

void min_across_planes(ArrayRef<f32> src, usize npix, usize channels, MutArrayRef<f32> out) noexcept {
    for (usize i = 0; i < npix; ++i) {
        f32 m = src[i];
        for (usize p = 1; p < channels; ++p) {
            const f32 v = src[p * npix + i];
            if (v < m) m = v;
        }
        out[i] = m;
    }
}

struct Rect {
    usize r0, c0, r1, c1; // inclusive
};

Rect quadtree_decomp(ArrayRef<f32> dark, usize h, Rect q, i32 num_levels) noexcept {
    auto mean_of = [&](const Rect& box) noexcept -> f64 {
        if (box.r0 > box.r1 || box.c0 > box.c1) return -Limits<f64>::infinity();
        f64 s = 0.0;
        const usize n = (box.r1 - box.r0 + 1) * (box.c1 - box.c0 + 1);
        for (usize c = box.c0; c <= box.c1; ++c) {
            for (usize r = box.r0; r <= box.r1; ++r) {
                s += as<f64>(dark[c * h + r]);
            }
        }
        return s / as<f64>(n);
    };
    for (i32 lvl = 0; lvl < num_levels; ++lvl) {
        if (q.r0 >= q.r1 || q.c0 >= q.c1) break;
        const i64 rmid = as<i64>(round((as<f64>(q.r0) + as<f64>(q.r1)) / 2.0));
        const i64 cmid = as<i64>(round((as<f64>(q.c0) + as<f64>(q.c1)) / 2.0));
        const Rect q1{q.r0, q.c0, as<usize>(rmid), as<usize>(cmid)};
        const Rect q2{q.r0, as<usize>(cmid + 1), as<usize>(rmid), q.c1};
        const Rect q3{as<usize>(rmid + 1), q.c0, q.r1, as<usize>(cmid)};
        const Rect q4{as<usize>(rmid + 1), as<usize>(cmid + 1), q.r1, q.c1};
        const f64 m[4] = {mean_of(q1), mean_of(q2), mean_of(q3), mean_of(q4)};
        i32 best = 0;
        for (i32 k = 1; k < 4; ++k) {
            if (m[k] > m[best]) best = k;
        }
        q = (best == 0) ? q1 : (best == 1) ? q2 : (best == 2) ? q3 : q4;
    }
    return q;
}

void simpledcp_atm_light(ArrayRef<f32> img, usize h, usize w, usize channels, f32 atm[3],
                         Allocator* alloc) {
    const usize npix = h * w;
    Rect q{0, 0, h - 1, w - 1};
    if (h >= 64 && w >= 64) {
        const i32 win_size = as<i32>(ceil(as<f64>(min(h, w)) / 400.0 * 15.0));
        Array<f32> dark(npix, 0.0f, alloc);
        Array<f32> eroded(npix, 0.0f, alloc);
        min_across_planes(img, npix, channels, dark.as_mut_ref());
        morph_square_f32<true>(dark.as_ref(), h, w, win_size, eroded.as_mut_ref(), alloc);
        q = quadtree_decomp(eroded.as_ref(), h, q, 5);
    }
    f64 best_dist = Limits<f64>::infinity();
    usize best_r = q.r0;
    usize best_c = q.c0;
    for (usize c = q.c0; c <= q.c1; ++c) {
        for (usize r = q.r0; r <= q.r1; ++r) {
            f64 s = 0.0;
            for (usize p = 0; p < channels; ++p) {
                const f64 v = as<f64>(img[p * npix + c * h + r]);
                s += abs(1.0 - v);
            }
            if (s < best_dist) {
                best_dist = s;
                best_r = r;
                best_c = c;
            }
        }
    }
    for (usize p = 0; p < channels; ++p) {
        atm[p] = img[p * npix + best_c * h + best_r];
    }
}

void approxdcp_atm_light(ArrayRef<f32> img, ArrayRef<f32> dark, usize npix, usize channels,
                         f32 atm[3], Allocator* alloc) {
    constexpr i32 kBins = 256;
    usize hist[kBins]{};
    for (usize i = 0; i < npix; ++i) {
        const f32 dv = is_nan(as<f64>(dark[i])) ? 0.0f : clamp(dark[i], 0.0f, 1.0f);
        i32 b = as<i32>(floor(dv * as<f32>(kBins - 1) + 0.5f));
        if (b < 0) b = 0;
        if (b >= kBins) b = kBins - 1;
        hist[b]++;
    }
    const f64 total = as<f64>(npix);
    f64 cum = 0.0;
    i32 bin_idx = kBins - 1;
    for (i32 b = 0; b < kBins; ++b) {
        cum += as<f64>(hist[b]);
        if (cum / total >= 0.999) {
            bin_idx = b;
            break;
        }
    }
    const f32 bin_w = 1.0f / as<f32>(kBins - 1);
    const f32 cutoff = as<f32>(bin_idx) / as<f32>(kBins - 1) - bin_w * 0.5f;

    Array<f32> gray(alloc);
    const f32* gv = img.data();
    if (channels == 3) {
        gray.resize(npix, 0.0f);
        for (usize i = 0; i < npix; ++i) {
            gray[i] = as<f32>(rgb_to_gray(as<f64>(img[i]), as<f64>(img[npix + i]),
                                          as<f64>(img[2 * npix + i])));
        }
        gv = gray.data();
    }

    f32 best_gray = -Limits<f32>::infinity();
    i64 best_idx = -1;
    for (usize i = 0; i < npix; ++i) {
        if (dark[i] >= cutoff) {
            if (gv[i] > best_gray) {
                best_gray = gv[i];
                best_idx = as<i64>(i);
            }
        }
    }
    if (best_idx < 0) {
        usize mi = 0;
        f32 mv = dark[0];
        for (usize i = 1; i < npix; ++i) {
            if (dark[i] > mv) {
                mv = dark[i];
                mi = i;
            }
        }
        for (usize p = 0; p < channels; ++p) {
            f32 v = img[p * npix + mi];
            if (v == 0.0f) v = Limits<f32>::kEpsilon;
            atm[p] = v;
        }
    } else {
        const usize bi = as<usize>(best_idx);
        for (usize p = 0; p < channels; ++p) {
            f32 v = img[p * npix + bi];
            if (v == 0.0f) v = Limits<f32>::kEpsilon;
            atm[p] = v;
        }
    }
}

void stretchlim_plane_f32(ArrayRef<f32> plane, f64 low_tol, f64 high_tol, f64& lo_out,
                          f64& hi_out, Allocator* alloc) {
    constexpr i32 kBins = 65536;
    Array<usize> hist(as<usize>(kBins), usize{0}, alloc);
    const usize n = plane.size();
    for (usize i = 0; i < n; ++i) {
        i32 bin = as<i32>(round(as<f64>(plane[i]) * as<f64>(kBins - 1)));
        if (bin < 0) bin = 0;
        if (bin >= kBins) bin = kBins - 1;
        hist[as<usize>(bin)]++;
    }
    const f64 nf = as<f64>(n);
    usize cum = 0;
    i32 low = kBins - 1;
    i32 high = kBins - 1;
    bool found_low = false;
    bool found_high = false;
    for (i32 k = 0; k < kBins; ++k) {
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
    if (high <= low) high = min(kBins - 1, low + 1);
    lo_out = as<f64>(low) / as<f64>(kBins - 1);
    hi_out = as<f64>(high) / as<f64>(kBins - 1);
}

void global_stretching_inplace(MutArrayRef<f32> img, usize npix, usize channels,
                               Allocator* alloc) {
    const usize total = npix * channels;
    f32 gmin = Limits<f32>::infinity();
    f32 gmax = -Limits<f32>::infinity();
    for (usize i = 0; i < total; ++i) {
        const f32 v = pow(img[i], 0.75f);
        img[i] = v;
        if (v < gmin) gmin = v;
        if (v > gmax) gmax = v;
    }
    const f64 span = as<f64>(gmax) - as<f64>(gmin);
    if (span > 0.0) {
        const f64 inv_span = 1.0 / span;
        for (usize i = 0; i < total; ++i) {
            f64 v = (as<f64>(img[i]) - as<f64>(gmin)) * inv_span;
            if (v < 0.0) v = 0.0;
            if (v > 1.0) v = 1.0;
            img[i] = as<f32>(v);
        }
    } else {
        for (usize i = 0; i < total; ++i) img[i] = 0.0f;
    }

    f64 lo_per[3]{};
    f64 hi_per[3]{};
    f64 sum_lo = 0.0;
    f64 sum_hi = 0.0;
    for (usize p = 0; p < channels; ++p) {
        stretchlim_plane_f32(ArrayRef<f32>(img.data() + p * npix, npix), 0.001, 0.999,
                             lo_per[p], hi_per[p], alloc);
        sum_lo += lo_per[p];
        sum_hi += hi_per[p];
    }
    const f64 mean_lo = sum_lo / as<f64>(channels);
    const f64 mean_hi = sum_hi / as<f64>(channels);
    constexpr f64 kAlpha = 0.8;

    for (usize p = 0; p < channels; ++p) {
        const f64 lo_n = lo_per[p] + kAlpha * (max(lo_per[p], mean_lo) - lo_per[p]);
        const f64 hi_n = hi_per[p] + kAlpha * (max(hi_per[p], mean_hi) - hi_per[p]);
        const f64 range = hi_n - lo_n;
        const f64 inv_range = (range != 0.0) ? (1.0 / range) : 0.0;
        f32* plane = img.data() + p * npix;
        for (usize i = 0; i < npix; ++i) {
            f64 t = (as<f64>(plane[i]) - lo_n) * inv_range;
            if (t < 0.0) t = 0.0;
            if (t > 1.0) t = 1.0;
            plane[i] = as<f32>(t);
        }
    }
}

} // namespace

ImageStatus box_filter_2d_replicate(ArrayRef<f64> in_col_major, usize height, usize width,
                                    i32 filter_size, MutArrayRef<f64> out_col_major,
                                    Allocator* alloc) {
    const usize n = height * width;
    if (in_col_major.size() != n || out_col_major.size() != n) {
        return ImageStatus::kInvalidArgument;
    }
    if (height == 0 || width == 0) return ImageStatus::kOk;
    if (filter_size <= 0) filter_size = 3;
    if ((filter_size % 2) == 0) filter_size += 1;

    const i32 k = filter_size;
    const i32 hk = (k - 1) / 2;
    const i32 ih = as<i32>(height);
    const i32 iw = as<i32>(width);
    const i32 hp = ih + 2 * hk;
    const i32 wp = iw + 2 * hk;
    const usize hs = as<usize>(hp + 1);
    const usize ws = as<usize>(wp + 1);

    Array<f64> sat(hs * ws, 0.0, alloc);
    auto at = [&](usize r, usize c) noexcept -> f64& { return sat[c * hs + r]; };

    for (i32 cp = 0; cp < wp; ++cp) {
        const i32 src_c = clamp(cp - hk, i32{0}, iw - 1);
        const usize src_col = as<usize>(src_c) * height;
        f64 col_run = 0.0;
        for (i32 rp = 0; rp < hp; ++rp) {
            const i32 src_r = clamp(rp - hk, i32{0}, ih - 1);
            col_run += in_col_major[src_col + as<usize>(src_r)];
            at(as<usize>(rp + 1), as<usize>(cp + 1)) =
                at(as<usize>(rp + 1), as<usize>(cp)) + col_run;
        }
    }

    const f64 inv_area = 1.0 / (as<f64>(k) * as<f64>(k));
    for (i32 c = 0; c < iw; ++c) {
        const usize c0 = as<usize>(c);
        const usize c1 = as<usize>(c + k);
        const usize dst_col = as<usize>(c) * height;
        for (i32 r = 0; r < ih; ++r) {
            const usize r0 = as<usize>(r);
            const usize r1 = as<usize>(r + k);
            const f64 sum = at(r1, c1) - at(r0, c1) - at(r1, c0) + at(r0, c0);
            out_col_major[dst_col + as<usize>(r)] = sum * inv_area;
        }
    }
    return ImageStatus::kOk;
}

ImageStatus guided_filter_gray(ArrayRef<f64> img_col_major, ArrayRef<f64> guide_col_major,
                               usize height, usize width, i32 nhood, f64 eps,
                               MutArrayRef<f64> out_col_major, Allocator* alloc) {
    const usize n = height * width;
    if (img_col_major.size() != n || out_col_major.size() != n) {
        return ImageStatus::kInvalidArgument;
    }
    if (!guide_col_major.empty() && guide_col_major.size() != n) {
        return ImageStatus::kInvalidArgument;
    }
    if (nhood <= 0 || (nhood % 2) == 0 || !is_finite(eps) || eps < 0.0) {
        return ImageStatus::kInvalidArgument;
    }
    if (height == 0 || width == 0) return ImageStatus::kOk;

    ArrayRef<f64> g = guide_col_major.empty() ? img_col_major : guide_col_major;
    Array<f64> ii(n, 0.0, alloc);
    Array<f64> ip(n, 0.0, alloc);
    for (usize i = 0; i < n; ++i) {
        ii[i] = g[i] * g[i];
        ip[i] = g[i] * img_col_major[i];
    }

    Array<f64> mean_i(n, 0.0, alloc);
    Array<f64> mean_p(n, 0.0, alloc);
    Array<f64> corr_i(n, 0.0, alloc);
    Array<f64> corr_ip(n, 0.0, alloc);
    (void)box_filter_2d_replicate(g, height, width, nhood, mean_i.as_mut_ref(), alloc);
    (void)box_filter_2d_replicate(img_col_major, height, width, nhood, mean_p.as_mut_ref(), alloc);
    (void)box_filter_2d_replicate(ii.as_ref(), height, width, nhood, corr_i.as_mut_ref(), alloc);
    (void)box_filter_2d_replicate(ip.as_ref(), height, width, nhood, corr_ip.as_mut_ref(), alloc);

    Array<f64> a_buf(n, 0.0, alloc);
    Array<f64> b_buf(n, 0.0, alloc);
    for (usize i = 0; i < n; ++i) {
        const f64 m_i = mean_i[i];
        const f64 m_p = mean_p[i];
        const f64 var_i = corr_i[i] - m_i * m_i;
        const f64 cov_ip = corr_ip[i] - m_i * m_p;
        const f64 denom = var_i + eps;
        const f64 a = (denom != 0.0) ? (cov_ip / denom) : 0.0;
        a_buf[i] = a;
        b_buf[i] = m_p - a * m_i;
    }

    Array<f64> mean_a(n, 0.0, alloc);
    Array<f64> mean_b(n, 0.0, alloc);
    (void)box_filter_2d_replicate(a_buf.as_ref(), height, width, nhood, mean_a.as_mut_ref(), alloc);
    (void)box_filter_2d_replicate(b_buf.as_ref(), height, width, nhood, mean_b.as_mut_ref(), alloc);

    for (usize i = 0; i < n; ++i) {
        out_col_major[i] = mean_a[i] * g[i] + mean_b[i];
    }
    return ImageStatus::kOk;
}

ImageStatus dehaze(ArrayRef<f32> in_planar_col_major, usize height, usize width, usize channels,
                   f64 amount, DehazeMethod method, ArrayRef<f32> atm_override,
                   DehazeContrast contrast, f64 boost_amount,
                   MutArrayRef<f32> out_planar_col_major,
                   MutArrayRef<f64> out_thickness_col_major, MutArrayRef<f64> out_atm,
                   Allocator* alloc) {
    if (height == 0 || width == 0 || (channels != 1 && channels != 3)) {
        return ImageStatus::kInvalidArgument;
    }
    const usize npix = height * width;
    if (in_planar_col_major.size() != npix * channels ||
        out_planar_col_major.size() != npix * channels) {
        return ImageStatus::kInvalidArgument;
    }
    if (!is_finite(amount) || amount < 0.0 || amount > 1.0 || !is_finite(boost_amount)) {
        return ImageStatus::kInvalidArgument;
    }
    if (!atm_override.empty()) {
        if (atm_override.size() != channels) return ImageStatus::kInvalidArgument;
        for (usize p = 0; p < channels; ++p) {
            if (!is_finite(as<f64>(atm_override[p])) || atm_override[p] < 0.0f) {
                return ImageStatus::kInvalidArgument;
            }
        }
    }
    if (!out_thickness_col_major.empty() && out_thickness_col_major.size() != npix) {
        return ImageStatus::kInvalidArgument;
    }
    if (!out_atm.empty() && out_atm.size() != channels) {
        return ImageStatus::kInvalidArgument;
    }

    // Clamp working copy to [0, 1].
    Array<f32> img(npix * channels, 0.0f, alloc);
    for (usize i = 0; i < npix * channels; ++i) {
        const f32 v = in_planar_col_major[i];
        img[i] = is_nan(as<f64>(v)) ? 0.0f : clamp(v, 0.0f, 1.0f);
    }

    f32 atm[3] = {0.0f, 0.0f, 0.0f};
    const bool has_atm = !atm_override.empty();
    if (has_atm) {
        for (usize p = 0; p < channels; ++p) atm[p] = atm_override[p];
    }

    i32 patch_size = as<i32>(ceil(as<f64>(min(height, width)) / 400.0 * 15.0));
    if (patch_size < 1) patch_size = 1;

    Array<f32> tmap(npix, 0.0f, alloc);
    f64 omega = 0.9;

    if (method == DehazeMethod::kSimpleDcp) {
        if (!has_atm) simpledcp_atm_light(img.as_ref(), height, width, channels, atm, alloc);
        Array<f32> mp(npix, 0.0f, alloc);
        min_across_planes(img.as_ref(), npix, channels, mp.as_mut_ref());
        Array<f64> t_in(npix, 0.0, alloc);
        Array<f64> t_out(npix, 0.0, alloc);
        for (usize i = 0; i < npix; ++i) t_in[i] = as<f64>(1.0f - mp[i]);
        (void)guided_filter_gray(t_in.as_ref(), t_in.as_ref(), height, width, 5, 0.01,
                                 t_out.as_mut_ref(), alloc);
        for (usize i = 0; i < npix; ++i) {
            tmap[i] = clamp(as<f32>(t_out[i]), 0.0f, 1.0f);
        }
        omega = 0.9;
    } else {
        Array<f32> dc_base(npix, 0.0f, alloc);
        Array<f32> dark(npix, 0.0f, alloc);
        min_across_planes(img.as_ref(), npix, channels, dc_base.as_mut_ref());
        morph_square_f32<true>(dc_base.as_ref(), height, width, patch_size, dark.as_mut_ref(),
                               alloc);
        if (!has_atm) approxdcp_atm_light(img.as_ref(), dark.as_ref(), npix, channels, atm, alloc);

        Array<f32> norm_i(npix, 0.0f, alloc);
        for (usize i = 0; i < npix; ++i) {
            f32 m = Limits<f32>::infinity();
            for (usize p = 0; p < channels; ++p) {
                const f32 a = atm[p];
                const f32 v = (a == 0.0f) ? Limits<f32>::infinity() : (img[p * npix + i] / a);
                if (v < m) m = v;
            }
            norm_i[i] = m;
        }
        // imopen(norm_i, se) = imdilate(imerode(norm_i, se), se)
        Array<f32> eroded(npix, 0.0f, alloc);
        Array<f32> opened(npix, 0.0f, alloc);
        morph_square_f32<true>(norm_i.as_ref(), height, width, patch_size, eroded.as_mut_ref(),
                               alloc);
        morph_square_f32<false>(eroded.as_ref(), height, width, patch_size, opened.as_mut_ref(),
                                alloc);

        Array<f64> t_in(npix, 0.0, alloc);
        Array<f64> guide(npix, 0.0, alloc);
        Array<f64> t_out(npix, 0.0, alloc);
        for (usize i = 0; i < npix; ++i) {
            t_in[i] = as<f64>(1.0f - opened[i]);
            if (channels == 3) {
                guide[i] = as<f64>(as<f32>(rgb_to_gray(
                    as<f64>(img[i]), as<f64>(img[npix + i]), as<f64>(img[2 * npix + i]))));
            } else {
                guide[i] = as<f64>(img[i]);
            }
        }
        i32 radius = as<i32>(ceil(as<f64>(min(height, width)) / 50.0));
        if (radius < 1) radius = 1;
        const i32 nhood = 2 * radius + 1;
        (void)guided_filter_gray(t_in.as_ref(), guide.as_ref(), height, width, nhood, 1e-4,
                                 t_out.as_mut_ref(), alloc);
        for (usize i = 0; i < npix; ++i) {
            tmap[i] = clamp(as<f32>(t_out[i]), 0.0f, 1.0f);
        }
        omega = 0.95;
    }

    // Thickness map T = 1 - tmap.
    Array<f64> thick(npix, 0.0, alloc);
    for (usize i = 0; i < npix; ++i) {
        const f64 tv = 1.0 - as<f64>(tmap[i]);
        thick[i] = tv;
        if (!out_thickness_col_major.empty()) out_thickness_col_major[i] = tv;
    }
    if (!out_atm.empty()) {
        for (usize p = 0; p < channels; ++p) out_atm[p] = as<f64>(atm[p]);
    }

    // Recover radiance.
    constexpr f32 kT0 = 0.1f;
    const f32 famount = as<f32>(amount);
    const f32 fomega = as<f32>(omega);
    for (usize i = 0; i < npix; ++i) {
        const f32 t = tmap[i];
        const f32 tb = 1.0f - fomega * (1.0f - t);
        const f32 tt = (tb > kT0) ? tb : kT0;
        const f32 tnew = min(1.0f, tb + famount);
        for (usize p = 0; p < channels; ++p) {
            const f32 a = atm[p];
            const f32 v = img[p * npix + i];
            const f32 rad = clamp(a + (v - a) / tt, 0.0f, 1.0f);
            out_planar_col_major[p * npix + i] = clamp(rad * tnew + a * (1.0f - tnew), 0.0f, 1.0f);
        }
    }

    // Post-processing.
    if (contrast == DehazeContrast::kGlobal) {
        global_stretching_inplace(out_planar_col_major, npix, channels, alloc);
    } else if (contrast == DehazeContrast::kBoost) {
        const f32 fboost = as<f32>((boost_amount > 0.0) ? boost_amount : 0.1);
        for (usize i = 0; i < npix; ++i) {
            const f32 g = 1.0f + famount * fboost * as<f32>(thick[i]);
            for (usize p = 0; p < channels; ++p) {
                out_planar_col_major[p * npix + i] =
                    clamp(out_planar_col_major[p * npix + i] * g, 0.0f, 1.0f);
            }
        }
    }

    return ImageStatus::kOk;
}

} // namespace keel::image
