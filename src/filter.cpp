// keelcpp/image/src/filter.cpp
// Pure zero-STL 2D/3D spatial, statistical, frequency, and denoising filters.

#include "keel/image.h"

#include "keel/algo.h"
#include "keel/arena.h"
#include "keel/array.h"
#include "keel/fft.h"
#include "keel/math.h"

namespace keel::image {

namespace {

constexpr f64 kPi = 3.14159265358979323846;

inline i32 fold_index(i32 i, i32 n, PadMode m) noexcept {
    if (n <= 0) {
        return 0;
    }
    if (i >= 0 && i < n) {
        return i;
    }
    switch (m) {
        case PadMode::kReplicate:
            return clamp(i, 0, n - 1);
        case PadMode::kSymmetric: {
            const i32 period = 2 * n;
            i32 j = i % period;
            if (j < 0) {
                j += period;
            }
            if (j >= n) {
                j = period - 1 - j;
            }
            return j;
        }
        case PadMode::kCircular: {
            i32 j = i % n;
            if (j < 0) {
                j += n;
            }
            return j;
        }
        case PadMode::kConstant:
            return clamp(i, 0, n - 1);
    }
    return clamp(i, 0, n - 1);
}

inline f64 sample_padded_2d(ArrayRef<f64> img, i32 r, i32 c, i32 h, i32 w, PadMode mode,
                            f64 pv) noexcept {
    if (mode == PadMode::kConstant) {
        if (r < 0 || r >= h || c < 0 || c >= w) {
            return pv;
        }
    } else {
        r = fold_index(r, h, mode);
        c = fold_index(c, w, mode);
    }
    return img[as<usize>(c) * as<usize>(h) + as<usize>(r)];
}

inline f64 sample_sym_2d(ArrayRef<f64> img, i32 r, i32 c, i32 h, i32 w) noexcept {
    r = fold_index(r, h, PadMode::kSymmetric);
    c = fold_index(c, w, PadMode::kSymmetric);
    return img[as<usize>(c) * as<usize>(h) + as<usize>(r)];
}

inline f64 med9(f64 p[9]) noexcept {
    auto pix_sort = [&](usize a, usize b) noexcept {
        if (p[a] > p[b]) {
            const f64 t = p[a];
            p[a] = p[b];
            p[b] = t;
        }
    };
    pix_sort(1, 2);
    pix_sort(4, 5);
    pix_sort(7, 8);
    pix_sort(0, 1);
    pix_sort(3, 4);
    pix_sort(6, 7);
    pix_sort(1, 2);
    pix_sort(4, 5);
    pix_sort(7, 8);
    pix_sort(0, 3);
    pix_sort(5, 8);
    pix_sort(4, 7);
    pix_sort(3, 6);
    pix_sort(1, 4);
    pix_sort(2, 5);
    pix_sort(4, 7);
    pix_sort(4, 2);
    pix_sort(6, 4);
    pix_sort(4, 2);
    return p[4];
}

// In-place quickselect: puts the k-th smallest element at `data[k]` and partitions
// `data[0..k-1] <= data[k] <= data[k+1..n-1]`.
void nth_select_f64(f64* data, usize n, usize k) noexcept {
    if (n <= 1 || k >= n) {
        return;
    }
    usize lo = 0;
    usize hi = n - 1;
    while (lo < hi) {
        if (hi - lo <= 16) {
            for (usize i = lo + 1; i <= hi; ++i) {
                const f64 v = data[i];
                usize j = i;
                while (j > lo && data[j - 1] > v) {
                    data[j] = data[j - 1];
                    --j;
                }
                data[j] = v;
            }
            return;
        }
        const usize mid = lo + (hi - lo) / 2;
        if (data[mid] < data[lo]) {
            swap(data[mid], data[lo]);
        }
        if (data[hi] < data[lo]) {
            swap(data[hi], data[lo]);
        }
        if (data[hi] < data[mid]) {
            swap(data[hi], data[mid]);
        }
        const f64 pivot = data[mid];
        swap(data[mid], data[hi - 1]);
        usize i = lo;
        usize j = hi - 1;
        for (;;) {
            while (++i < hi - 1 && data[i] < pivot) {
            }
            while (--j > lo && data[j] > pivot) {
            }
            if (i >= j) {
                break;
            }
            swap(data[i], data[j]);
        }
        swap(data[i], data[hi - 1]);
        if (k == i) {
            return;
        }
        if (k < i) {
            hi = i - 1;
        } else {
            lo = i + 1;
        }
    }
}

f64 median_of_buffer(f64* data, usize n) noexcept {
    if (n == 0) {
        return 0.0;
    }
    const usize mid = n / 2;
    nth_select_f64(data, n, mid);
    f64 med = data[mid];
    if ((n & 1) == 0) {
        f64 left_max = -Limits<f64>::infinity();
        for (usize i = 0; i < mid; ++i) {
            if (data[i] > left_max) {
                left_max = data[i];
            }
        }
        med = 0.5 * (left_max + med);
    }
    return med;
}

Array<f64> gauss1d_kernel(f64 sigma, Allocator* alloc) {
    if (!(sigma > 0.0)) {
        sigma = 0.5;
    }
    i32 half = as<i32>(ceil(2.0 * sigma));
    if (half < 1) {
        half = 1;
    }
    const i32 k_len = 2 * half + 1;
    Array<f64> k(as<usize>(k_len), 0.0, alloc);
    f64 sum = 0.0;
    const f64 inv2s2 = 0.5 / (sigma * sigma);
    for (i32 i = -half; i <= half; ++i) {
        const f64 w = exp(-as<f64>(i * i) * inv2s2);
        k[as<usize>(i + half)] = w;
        sum += w;
    }
    const f64 inv = 1.0 / sum;
    for (usize i = 0; i < k.size(); ++i) {
        k[i] *= inv;
    }
    return k;
}

struct Off2D {
    i32 dr;
    i32 dc;
};

void build_domain_offsets(ArrayRef<u8> domain, usize dom_h, usize dom_w, Array<Off2D>& offs) {
    offs.clear();
    offs.reserve(dom_h * dom_w);
    const i32 rh = as<i32>(dom_h) / 2;
    const i32 ch = as<i32>(dom_w) / 2;
    for (usize c = 0; c < dom_w; ++c) {
        for (usize r = 0; r < dom_h; ++r) {
            if (domain[c * dom_h + r] != 0) {
                offs.push_back(Off2D{as<i32>(r) - rh, as<i32>(c) - ch});
            }
        }
    }
}

} // namespace

ImageStatus padarray_f64(ArrayRef<f64> src_col_major, usize height, usize width, usize pages,
                         i32 pre_r, i32 post_r, i32 pre_c, i32 post_c, i32 pre_p, i32 post_p,
                         PadMode mode, f64 pad_value, MutArrayRef<f64> out_col_major) noexcept {
    if (pre_r < 0 || post_r < 0 || pre_c < 0 || post_c < 0 || pre_p < 0 || post_p < 0) {
        return ImageStatus::kInvalidArgument;
    }
    const i32 h = as<i32>(height);
    const i32 w = as<i32>(width);
    const i32 p = as<i32>(pages);
    const i32 h2 = h + pre_r + post_r;
    const i32 w2 = w + pre_c + post_c;
    const i32 p2 = p + pre_p + post_p;
    if (src_col_major.size() < height * width * pages ||
        out_col_major.size() < as<usize>(h2) * as<usize>(w2) * as<usize>(p2)) {
        return ImageStatus::kInvalidArgument;
    }
    if (h2 == 0 || w2 == 0 || p2 == 0) {
        return ImageStatus::kOk;
    }
    if (h == 0 || w == 0 || p == 0) {
        if (mode != PadMode::kConstant) {
            return ImageStatus::kInvalidArgument;
        }
        const usize out_n = as<usize>(h2) * as<usize>(w2) * as<usize>(p2);
        for (usize i = 0; i < out_n; ++i) {
            out_col_major[i] = pad_value;
        }
        return ImageStatus::kOk;
    }

    const usize plane = height * width;
    const usize plane2 = as<usize>(h2) * as<usize>(w2);

    for (i32 pi = 0; pi < p2; ++pi) {
        const i32 p_src_raw = pi - pre_p;
        for (i32 ci = 0; ci < w2; ++ci) {
            const i32 c_src_raw = ci - pre_c;
            for (i32 ri = 0; ri < h2; ++ri) {
                const i32 r_src_raw = ri - pre_r;
                f64 v = pad_value;
                if (mode == PadMode::kConstant) {
                    if (r_src_raw >= 0 && r_src_raw < h && c_src_raw >= 0 && c_src_raw < w &&
                        p_src_raw >= 0 && p_src_raw < p) {
                        v = src_col_major[as<usize>(p_src_raw) * plane +
                                          as<usize>(c_src_raw) * height + as<usize>(r_src_raw)];
                    }
                } else {
                    const i32 rs = fold_index(r_src_raw, h, mode);
                    const i32 cs = fold_index(c_src_raw, w, mode);
                    const i32 ps = fold_index(p_src_raw, p, mode);
                    v = src_col_major[as<usize>(ps) * plane + as<usize>(cs) * height +
                                      as<usize>(rs)];
                }
                out_col_major[as<usize>(pi) * plane2 + as<usize>(ci) * as<usize>(h2) +
                              as<usize>(ri)] = v;
            }
        }
    }
    return ImageStatus::kOk;
}

ImageStatus fspecial_average(i32 rows, i32 cols, Kernel2D& out, Allocator* alloc) {
    if (rows <= 0 || cols <= 0) {
        return ImageStatus::kInvalidArgument;
    }
    const usize r = as<usize>(rows);
    const usize c = as<usize>(cols);
    const f64 w = 1.0 / (as<f64>(rows) * as<f64>(cols));
    out.rows = r;
    out.cols = c;
    out.data_col_major = Array<f64>(r * c, w, alloc);
    return ImageStatus::kOk;
}

ImageStatus fspecial_gaussian(i32 rows, i32 cols, f64 sigma, Kernel2D& out, Allocator* alloc) {
    if (rows <= 0 || cols <= 0 || !is_finite(sigma) || !(sigma > 0.0)) {
        return ImageStatus::kInvalidArgument;
    }
    const usize ru = as<usize>(rows);
    const usize cu = as<usize>(cols);
    out.rows = ru;
    out.cols = cu;
    out.data_col_major = Array<f64>(ru * cu, 0.0, alloc);

    const f64 cy = as<f64>(rows - 1) * 0.5;
    const f64 cx = as<f64>(cols - 1) * 0.5;
    const f64 inv2s2 = 1.0 / (2.0 * sigma * sigma);
    f64 sum = 0.0;
    for (i32 c = 0; c < cols; ++c) {
        const f64 dx = as<f64>(c) - cx;
        for (i32 r = 0; r < rows; ++r) {
            const f64 dy = as<f64>(r) - cy;
            const f64 v = exp(-(dx * dx + dy * dy) * inv2s2);
            out.data_col_major[as<usize>(c) * ru + as<usize>(r)] = v;
            sum += v;
        }
    }
    if (sum > 0.0) {
        for (usize i = 0; i < out.data_col_major.size(); ++i) {
            out.data_col_major[i] /= sum;
        }
    }
    return ImageStatus::kOk;
}

ImageStatus fspecial_laplacian(f64 alpha, Kernel2D& out, Allocator* alloc) {
    if (!is_finite(alpha)) {
        return ImageStatus::kInvalidArgument;
    }
    alpha = clamp(alpha, 0.0, 1.0);
    const f64 s = 1.0 / (alpha + 1.0);
    out.rows = 3;
    out.cols = 3;
    out.data_col_major = Array<f64>(9, 0.0, alloc);
    const f64 k[9] = {
        s * alpha / 4.0, s * (1.0 - alpha) / 4.0, s * alpha / 4.0,
        s * (1.0 - alpha) / 4.0, -s, s * (1.0 - alpha) / 4.0,
        s * alpha / 4.0, s * (1.0 - alpha) / 4.0, s * alpha / 4.0,
    };
    for (usize i = 0; i < 9; ++i) {
        out.data_col_major[i] = k[i];
    }
    return ImageStatus::kOk;
}

ImageStatus fspecial_log(i32 rows, i32 cols, f64 sigma, Kernel2D& out, Allocator* alloc) {
    if (rows <= 0 || cols <= 0 || !is_finite(sigma) || !(sigma > 0.0)) {
        return ImageStatus::kInvalidArgument;
    }
    const usize ru = as<usize>(rows);
    const usize cu = as<usize>(cols);
    const usize n = ru * cu;
    out.rows = ru;
    out.cols = cu;
    out.data_col_major = Array<f64>(n, 0.0, alloc);

    ArenaTemp scratch = get_scratch(alloc);
    Array<f64> g(n, 0.0, scratch.allocator());

    const f64 cy = as<f64>(rows - 1) * 0.5;
    const f64 cx = as<f64>(cols - 1) * 0.5;
    const f64 s2 = sigma * sigma;
    f64 gsum = 0.0;
    for (i32 c = 0; c < cols; ++c) {
        const f64 dx = as<f64>(c) - cx;
        for (i32 r = 0; r < rows; ++r) {
            const f64 dy = as<f64>(r) - cy;
            const f64 e = exp(-(dx * dx + dy * dy) / (2.0 * s2));
            g[as<usize>(c) * ru + as<usize>(r)] = e;
            gsum += e;
        }
    }
    if (gsum > 0.0) {
        for (usize i = 0; i < n; ++i) {
            g[i] /= gsum;
        }
    }
    f64 mean = 0.0;
    for (i32 c = 0; c < cols; ++c) {
        const f64 dx = as<f64>(c) - cx;
        for (i32 r = 0; r < rows; ++r) {
            const f64 dy = as<f64>(r) - cy;
            const f64 rsq = dx * dx + dy * dy;
            const usize idx = as<usize>(c) * ru + as<usize>(r);
            const f64 val = (rsq - 2.0 * s2) / (s2 * s2) * g[idx];
            out.data_col_major[idx] = val;
            mean += val;
        }
    }
    mean /= as<f64>(n);
    for (usize i = 0; i < n; ++i) {
        out.data_col_major[i] -= mean;
    }
    return ImageStatus::kOk;
}

ImageStatus fspecial_sobel(Kernel2D& out, Allocator* alloc) {
    out.rows = 3;
    out.cols = 3;
    out.data_col_major = Array<f64>(9, 0.0, alloc);
    const f64 k[9] = {1.0, 0.0, -1.0, 2.0, 0.0, -2.0, 1.0, 0.0, -1.0};
    for (usize i = 0; i < 9; ++i) {
        out.data_col_major[i] = k[i];
    }
    return ImageStatus::kOk;
}

ImageStatus fspecial_prewitt(Kernel2D& out, Allocator* alloc) {
    out.rows = 3;
    out.cols = 3;
    out.data_col_major = Array<f64>(9, 0.0, alloc);
    const f64 k[9] = {1.0, 0.0, -1.0, 1.0, 0.0, -1.0, 1.0, 0.0, -1.0};
    for (usize i = 0; i < 9; ++i) {
        out.data_col_major[i] = k[i];
    }
    return ImageStatus::kOk;
}

ImageStatus fspecial_disk(f64 radius, Kernel2D& out, Allocator* alloc) {
    if (!is_finite(radius) || !(radius > 0.0) || radius > 1e5) {
        return ImageStatus::kInvalidArgument;
    }
    const i32 crad = as<i32>(ceil(radius - 0.5));
    const i32 side = 2 * crad + 1;
    const usize su = as<usize>(side);
    const f64 r2 = radius * radius;
    out.rows = su;
    out.cols = su;
    out.data_col_major = Array<f64>(su * su, 0.0, alloc);

    auto clamp1 = [](f64 v) noexcept -> f64 { return clamp(v, -1.0, 1.0); };
    auto sq = [](f64 v) noexcept -> f64 { return v * v; };

    for (i32 ii = 0; ii < side; ++ii) {
        for (i32 jj = 0; jj < side; ++jj) {
            const f64 y = as<f64>(-crad + ii);
            const f64 x = as<f64>(-crad + jj);
            const f64 ax = abs(x);
            const f64 ay = abs(y);
            const f64 maxxy = max(ax, ay);
            const f64 minxy = min(ax, ay);
            const f64 m1 = (r2 < sq(maxxy + 0.5) + sq(minxy - 0.5))
                               ? (minxy - 0.5)
                               : sqrt(max(0.0, r2 - sq(maxxy + 0.5)));
            const f64 m2 = (r2 > sq(maxxy - 0.5) + sq(minxy + 0.5))
                               ? (minxy + 0.5)
                               : sqrt(max(0.0, r2 - sq(maxxy - 0.5)));
            const bool mask =
                ((r2 < sq(maxxy + 0.5) + sq(minxy + 0.5)) &&
                 (r2 > sq(maxxy - 0.5) + sq(minxy - 0.5))) ||
                ((minxy == 0.0) && (maxxy - 0.5 < radius) && (maxxy + 0.5 >= radius));
            f64 s = 0.0;
            if (mask) {
                const f64 a1 = asin(clamp1(m1 / radius));
                const f64 a2 = asin(clamp1(m2 / radius));
                s = r2 * (0.5 * (a2 - a1) + 0.25 * (sin(2.0 * a2) - sin(2.0 * a1))) -
                    (maxxy - 0.5) * (m2 - m1) + (m1 - minxy + 0.5);
            }
            if (sq(maxxy + 0.5) + sq(minxy + 0.5) < r2) {
                s += 1.0;
            }
            out.data_col_major[as<usize>(jj) * su + as<usize>(ii)] = s;
        }
    }

    auto at = [&](i32 i, i32 j) noexcept -> f64& {
        return out.data_col_major[as<usize>(j) * su + as<usize>(i)];
    };
    at(crad, crad) = min(kPi * r2, kPi * 0.5);
    if (crad > 0 && radius > as<f64>(crad) - 0.5 && r2 < sq(as<f64>(crad) - 0.5) + 0.25) {
        const f64 mm = sqrt(max(0.0, r2 - sq(as<f64>(crad) - 0.5)));
        const f64 mmn = mm / radius;
        const f64 am = asin(clamp1(mmn));
        const f64 sg0 = 2.0 * (r2 * (0.5 * am + 0.25 * sin(2.0 * am)) - mm * (as<f64>(crad) - 0.5));
        at(2 * crad, crad) = sg0;
        at(crad, 2 * crad) = sg0;
        at(crad, 0) = sg0;
        at(0, crad) = sg0;
        at(2 * crad - 1, crad) -= sg0;
        at(crad, 2 * crad - 1) -= sg0;
        at(crad, 1) -= sg0;
        at(1, crad) -= sg0;
    }
    at(crad, crad) = min(at(crad, crad), 1.0);

    f64 sum = 0.0;
    for (usize i = 0; i < out.data_col_major.size(); ++i) {
        sum += out.data_col_major[i];
    }
    if (sum > 0.0) {
        for (usize i = 0; i < out.data_col_major.size(); ++i) {
            out.data_col_major[i] /= sum;
        }
    }
    return ImageStatus::kOk;
}

ImageStatus fspecial_motion(f64 length, f64 theta_deg, Kernel2D& out, Allocator* alloc) {
    if (!is_finite(length) || !is_finite(theta_deg) || length > 1e5) {
        return ImageStatus::kInvalidArgument;
    }
    constexpr f64 kEps = Limits<f64>::kEpsilon;
    constexpr f64 kLineWdt = 1.0;
    const f64 len = max(1.0, length);
    const f64 half = (len - 1.0) * 0.5;
    f64 tmod = fmod(theta_deg, 180.0);
    if (tmod < 0.0) {
        tmod += 180.0;
    }
    const f64 phi = tmod / 180.0 * kPi;
    const f64 cosphi = cos(phi);
    const f64 sinphi = sin(phi);
    const f64 xsign = (cosphi > 0.0) ? 1.0 : (cosphi < 0.0 ? -1.0 : 0.0);

    const i64 sx = as<i64>(trunc(half * cosphi + kLineWdt * xsign - len * kEps));
    const i64 sy = as<i64>(trunc(half * sinphi + kLineWdt - len * kEps));
    const i32 ncol = as<i32>(abs(sx)) + 1;
    const i32 nrow = as<i32>(sy) + 1;

    ArenaTemp scratch = get_scratch(alloc);
    Array<f64> d_half(as<usize>(nrow) * as<usize>(ncol), 0.0, scratch.allocator());
    for (i32 r = 0; r < nrow; ++r) {
        for (i32 c = 0; c < ncol; ++c) {
            const f64 x = as<f64>(c) * xsign;
            const f64 y = as<f64>(r);
            f64 d = y * cosphi - x * sinphi;
            const f64 rad = hypot(x, y);
            if (rad >= half && abs(d) <= kLineWdt) {
                const f64 x2 = (cosphi != 0.0) ? (half - abs((x + d * sinphi) / cosphi)) : 0.0;
                d = sqrt(d * d + x2 * x2);
            }
            f64 w = kLineWdt + kEps - abs(d);
            if (w < 0.0) {
                w = 0.0;
            }
            d_half[as<usize>(r) * as<usize>(ncol) + as<usize>(c)] = w;
        }
    }

    const i32 h = 2 * nrow - 1;
    const i32 w = 2 * ncol - 1;
    const usize hu = as<usize>(h);
    const usize wu = as<usize>(w);
    Array<f64> hm(hu * wu, 0.0, scratch.allocator());
    auto dat = [&](i32 r, i32 c) noexcept -> f64 {
        return d_half[as<usize>(r) * as<usize>(ncol) + as<usize>(c)];
    };
    for (i32 r = 0; r < nrow; ++r) {
        for (i32 c = 0; c < ncol; ++c) {
            hm[as<usize>(r) * wu + as<usize>(c)] = dat(nrow - 1 - r, ncol - 1 - c);
        }
    }
    for (i32 r = 0; r < nrow; ++r) {
        for (i32 c = 0; c < ncol; ++c) {
            hm[as<usize>(nrow - 1 + r) * wu + as<usize>(ncol - 1 + c)] = dat(r, c);
        }
    }

    f64 s = 0.0;
    for (usize i = 0; i < hm.size(); ++i) {
        s += hm[i];
    }
    s += kEps * len * len;
    for (usize i = 0; i < hm.size(); ++i) {
        hm[i] /= s;
    }

    out.rows = hu;
    out.cols = wu;
    out.data_col_major = Array<f64>(hu * wu, 0.0, alloc);
    for (i32 c = 0; c < w; ++c) {
        for (i32 r = 0; r < h; ++r) {
            const i32 rr = (cosphi > 0.0) ? (h - 1 - r) : r;
            out.data_col_major[as<usize>(c) * hu + as<usize>(r)] =
                hm[as<usize>(rr) * wu + as<usize>(c)];
        }
    }
    return ImageStatus::kOk;
}

ImageStatus fspecial_unsharp(f64 alpha, Kernel2D& out, Allocator* alloc) {
    if (!(alpha >= 0.0 && alpha <= 1.0)) {
        return ImageStatus::kInvalidArgument;
    }
    const f64 d = alpha + 1.0;
    out.rows = 3;
    out.cols = 3;
    out.data_col_major = Array<f64>(9, 0.0, alloc);
    const f64 k[9] = {
        -alpha / d, (alpha - 1.0) / d, -alpha / d,
        (alpha - 1.0) / d, (alpha + 5.0) / d, (alpha - 1.0) / d,
        -alpha / d, (alpha - 1.0) / d, -alpha / d,
    };
    for (usize i = 0; i < 9; ++i) {
        out.data_col_major[i] = k[i];
    }
    return ImageStatus::kOk;
}

ImageStatus imfilter_2d(ArrayRef<f64> img_col_major, usize height, usize width,
                        ArrayRef<f64> kernel_col_major, usize k_h, usize k_w, PadMode boundary,
                        f64 pad_value, bool full, bool flip_kernel,
                        MutArrayRef<f64> out_col_major) noexcept {
    if (k_h == 0 || k_w == 0) {
        return ImageStatus::kInvalidArgument;
    }
    if (height == 0 || width == 0) {
        return ImageStatus::kOk;
    }
    const usize out_h = full ? (height + k_h - 1) : height;
    const usize out_w = full ? (width + k_w - 1) : width;
    if (img_col_major.size() < height * width || kernel_col_major.size() < k_h * k_w ||
        out_col_major.size() < out_h * out_w) {
        return ImageStatus::kInvalidArgument;
    }

    const i32 h = as<i32>(height);
    const i32 w = as<i32>(width);
    const i32 kh = as<i32>(k_h);
    const i32 kw = as<i32>(k_w);
    const i32 half_r = (kh - 1) / 2;
    const i32 half_c = (kw - 1) / 2;

    auto get_k = [&](i32 ki, i32 kj) noexcept -> f64 {
        const i32 kki = flip_kernel ? (kh - 1 - ki) : ki;
        const i32 kkj = flip_kernel ? (kw - 1 - kj) : kj;
        return kernel_col_major[as<usize>(kkj) * k_h + as<usize>(kki)];
    };

    if (full) {
        const i32 oh = as<i32>(out_h);
        const i32 ow = as<i32>(out_w);
        for (i32 oc = 0; oc < ow; ++oc) {
            for (i32 orow = 0; orow < oh; ++orow) {
                f64 s = 0.0;
                for (i32 kj = 0; kj < kw; ++kj) {
                    const i32 c_in = oc + kj - (kw - 1);
                    for (i32 ki = 0; ki < kh; ++ki) {
                        const i32 r_in = orow + ki - (kh - 1);
                        s += get_k(ki, kj) *
                             sample_padded_2d(img_col_major, r_in, c_in, h, w, boundary, pad_value);
                    }
                }
                out_col_major[as<usize>(oc) * out_h + as<usize>(orow)] = s;
            }
        }
        return ImageStatus::kOk;
    }

    const f64* src = img_col_major.data();
    f64* dst = out_col_major.data();
    const i32 ir0 = max(0, half_r);
    const i32 ir1 = min(h, h - kh + 1 + half_r);
    const i32 ic0 = max(0, half_c);
    const i32 ic1 = min(w, w - kw + 1 + half_c);

    if (ir1 > ir0 && ic1 > ic0) {
        for (i32 oc = ic0; oc < ic1; ++oc) {
            f64* dcol = dst + as<usize>(oc) * height;
            for (i32 r = ir0; r < ir1; ++r) {
                dcol[r] = 0.0;
            }
            for (i32 kj = 0; kj < kw; ++kj) {
                const i32 c_in = oc + kj - half_c;
                for (i32 ki = 0; ki < kh; ++ki) {
                    const f64 wt = get_k(ki, kj);
                    const f64* scol = src + as<usize>(c_in) * height + (ki - half_r);
                    for (i32 r = ir0; r < ir1; ++r) {
                        dcol[r] += wt * scol[r];
                    }
                }
            }
        }
    }

    for (i32 oc = 0; oc < w; ++oc) {
        const bool col_interior = (oc >= ic0 && oc < ic1);
        for (i32 orow = 0; orow < h; ++orow) {
            if (col_interior && orow >= ir0 && orow < ir1) {
                continue;
            }
            f64 s = 0.0;
            for (i32 kj = 0; kj < kw; ++kj) {
                const i32 c_in = oc + kj - half_c;
                for (i32 ki = 0; ki < kh; ++ki) {
                    const i32 r_in = orow + ki - half_r;
                    s += get_k(ki, kj) *
                         sample_padded_2d(img_col_major, r_in, c_in, h, w, boundary, pad_value);
                }
            }
            dst[as<usize>(oc) * height + as<usize>(orow)] = s;
        }
    }
    return ImageStatus::kOk;
}

ImageStatus box_filter_2d_u8_replicate(ArrayRef<u8> in_col_major, usize height, usize width,
                                       i32 filter_size, MutArrayRef<u8> out_col_major,
                                       Allocator* alloc) {
    const usize n = height * width;
    if (in_col_major.size() < n || out_col_major.size() < n) {
        return ImageStatus::kInvalidArgument;
    }
    if (n == 0) {
        return ImageStatus::kOk;
    }
    if (filter_size <= 0) {
        filter_size = 3;
    }
    if ((filter_size & 1) == 0) {
        filter_size += 1;
    }

    const i32 k = filter_size;
    const i32 hk = (k - 1) / 2;
    const i32 h = as<i32>(height);
    const i32 w = as<i32>(width);
    const i32 hs = h + 2 * hk + 1;
    const i32 ws = w + 2 * hk + 1;

    ArenaTemp scratch = get_scratch(alloc);
    if (as<u64>(hs) * as<u64>(ws) * u64{255} < (u64{1} << 32) &&
        as<u64>(k) * as<u64>(k) * u64{511} < (u64{1} << 32)) {
        Array<u32> sat(as<usize>(hs) * as<usize>(ws), u32{0}, scratch.allocator());
        auto q = [&](i32 r, i32 c) noexcept -> u32& {
            return sat[as<usize>(c) * as<usize>(hs) + as<usize>(r)];
        };
        for (i32 c = 1; c < ws; ++c) {
            const i32 sc = clamp((c - 1) - hk, 0, w - 1);
            const usize src_col = as<usize>(sc) * height;
            for (i32 r = 1; r < hs; ++r) {
                const i32 sr = clamp((r - 1) - hk, 0, h - 1);
                q(r, c) = as<u32>(in_col_major[src_col + as<usize>(sr)]) + q(r - 1, c) +
                          q(r, c - 1) - q(r - 1, c - 1);
            }
        }
        const u32 k2 = as<u32>(k) * as<u32>(k);
        for (i32 c = 0; c < w; ++c) {
            for (i32 r = 0; r < h; ++r) {
                const u32 sum = (q(r + k, c + k) + q(r, c)) - (q(r, c + k) + q(r + k, c));
                out_col_major[as<usize>(c) * height + as<usize>(r)] =
                    as<u8>((u32{2} * sum + k2) / (u32{2} * k2));
            }
        }
        return ImageStatus::kOk;
    }

    Array<f64> in_f(n, scratch.allocator());
    Array<f64> out_f(n, scratch.allocator());
    for (usize i = 0; i < n; ++i) {
        in_f[i] = as<f64>(in_col_major[i]);
    }
    const ImageStatus st =
        box_filter_2d_replicate(in_f, height, width, k, out_f, scratch.allocator());
    if (st != ImageStatus::kOk) {
        return st;
    }
    for (usize i = 0; i < n; ++i) {
        const f64 v = clamp(out_f[i], 0.0, 255.0);
        out_col_major[i] = as<u8>(round(v));
    }
    return ImageStatus::kOk;
}

ImageStatus integral_box_filter_2d(ArrayRef<f64> sat_col_major, usize h1, usize w1, usize pages,
                                   i32 fh, i32 fw, f64 norm_factor,
                                   MutArrayRef<f64> out_col_major) noexcept {
    if (fh <= 0 || fw <= 0 || (fh & 1) == 0 || (fw & 1) == 0 || h1 < 2 || w1 < 2 || pages == 0) {
        return ImageStatus::kInvalidArgument;
    }
    const usize h = h1 - 1;
    const usize w = w1 - 1;
    if (as<usize>(fh) > h || as<usize>(fw) > w) {
        return ImageStatus::kInvalidArgument;
    }
    const usize out_h = h - as<usize>(fh) + 1;
    const usize out_w = w - as<usize>(fw) + 1;
    const usize in_plane = h1 * w1;
    const usize out_plane = out_h * out_w;
    if (sat_col_major.size() < in_plane * pages || out_col_major.size() < out_plane * pages) {
        return ImageStatus::kInvalidArgument;
    }

    for (usize c = 0; c < pages; ++c) {
        const usize base = c * in_plane;
        const usize out_base = c * out_plane;
        for (usize oj = 0; oj < out_w; ++oj) {
            const usize c0 = oj;
            const usize c1 = oj + as<usize>(fw);
            for (usize oi = 0; oi < out_h; ++oi) {
                const usize r0 = oi;
                const usize r1 = oi + as<usize>(fh);
                const f64 s = sat_col_major[base + c1 * h1 + r1] -
                              sat_col_major[base + c1 * h1 + r0] -
                              sat_col_major[base + c0 * h1 + r1] +
                              sat_col_major[base + c0 * h1 + r0];
                out_col_major[out_base + oj * out_h + oi] = s * norm_factor;
            }
        }
    }
    return ImageStatus::kOk;
}

ImageStatus integral_box_filter_3d(ArrayRef<f64> sat_col_major, usize h1, usize w1, usize d1,
                                   i32 fh, i32 fw, i32 fp, f64 norm_factor,
                                   MutArrayRef<f64> out_col_major) noexcept {
    if (fh <= 0 || fw <= 0 || fp <= 0 || (fh & 1) == 0 || (fw & 1) == 0 || (fp & 1) == 0 ||
        h1 < 2 || w1 < 2 || d1 < 2) {
        return ImageStatus::kInvalidArgument;
    }
    const usize h = h1 - 1;
    const usize w = w1 - 1;
    const usize d = d1 - 1;
    if (as<usize>(fh) > h || as<usize>(fw) > w || as<usize>(fp) > d) {
        return ImageStatus::kInvalidArgument;
    }
    const usize out_h = h - as<usize>(fh) + 1;
    const usize out_w = w - as<usize>(fw) + 1;
    const usize out_d = d - as<usize>(fp) + 1;
    const usize plane = h1 * w1;
    if (sat_col_major.size() < plane * d1 || out_col_major.size() < out_h * out_w * out_d) {
        return ImageStatus::kInvalidArgument;
    }

    auto iat = [&](usize r, usize c, usize p) noexcept -> f64 {
        return sat_col_major[p * plane + c * h1 + r];
    };

    for (usize ok = 0; ok < out_d; ++ok) {
        const usize p0 = ok;
        const usize p1 = ok + as<usize>(fp);
        for (usize oj = 0; oj < out_w; ++oj) {
            const usize c0 = oj;
            const usize c1 = oj + as<usize>(fw);
            for (usize oi = 0; oi < out_h; ++oi) {
                const usize r0 = oi;
                const usize r1 = oi + as<usize>(fh);
                const f64 s = iat(r1, c1, p1) - iat(r0, c1, p1) - iat(r1, c0, p1) -
                              iat(r1, c1, p0) + iat(r0, c0, p1) + iat(r0, c1, p0) +
                              iat(r1, c0, p0) - iat(r0, c0, p0);
                out_col_major[ok * out_h * out_w + oj * out_h + oi] = s * norm_factor;
            }
        }
    }
    return ImageStatus::kOk;
}

ImageStatus modefilt_u8(ArrayRef<u8> src_col_major, usize height, usize width, usize pages, i32 fh,
                        i32 fw, i32 fd, PadMode pad_mode, MutArrayRef<u8> out_col_major) noexcept {
    const bool is_3d = (fd > 0);
    if (fh <= 0 || fw <= 0 || (fh & 1) == 0 || (fw & 1) == 0 || (is_3d && (fd & 1) == 0)) {
        return ImageStatus::kInvalidArgument;
    }
    const usize d_eff = is_3d ? pages : 1;
    const usize plane = height * width;
    const usize total = plane * d_eff;
    if (src_col_major.size() < total || out_col_major.size() < total) {
        return ImageStatus::kInvalidArgument;
    }
    if (total == 0) {
        return ImageStatus::kOk;
    }

    const i32 h = as<i32>(height);
    const i32 w = as<i32>(width);
    const i32 d = as<i32>(d_eff);
    const i32 hh = fh / 2;
    const i32 hw = fw / 2;
    const i32 hd = is_3d ? (fd / 2) : 0;

    auto reflect = [&](i32 v, i32 n) noexcept -> i32 {
        if (pad_mode == PadMode::kConstant) {
            return (v < 0 || v >= n) ? -1 : v;
        }
        return fold_index(v, n, pad_mode);
    };

    for (i32 p = 0; p < d; ++p) {
        for (i32 c = 0; c < w; ++c) {
            for (i32 r = 0; r < h; ++r) {
                i32 hist[256] = {};
                for (i32 dp = -hd; dp <= hd; ++dp) {
                    const i32 ps = is_3d ? reflect(p + dp, d) : 0;
                    for (i32 dc = -hw; dc <= hw; ++dc) {
                        const i32 cs = reflect(c + dc, w);
                        for (i32 dr = -hh; dr <= hh; ++dr) {
                            const i32 rs = reflect(r + dr, h);
                            u8 v = 0;
                            if (rs >= 0 && cs >= 0 && (!is_3d || ps >= 0)) {
                                v = src_col_major[as<usize>(ps) * plane + as<usize>(cs) * height +
                                                  as<usize>(rs)];
                            }
                            ++hist[v];
                        }
                    }
                }
                i32 best_val = 0;
                i32 best_cnt = -1;
                for (i32 v = 0; v < 256; ++v) {
                    if (hist[v] > best_cnt) {
                        best_cnt = hist[v];
                        best_val = v;
                    }
                }
                out_col_major[as<usize>(p) * plane + as<usize>(c) * height + as<usize>(r)] =
                    as<u8>(best_val);
            }
        }
    }
    return ImageStatus::kOk;
}

ImageStatus modefilt_f64(ArrayRef<f64> src_col_major, usize height, usize width, usize pages,
                         i32 fh, i32 fw, i32 fd, PadMode pad_mode, MutArrayRef<f64> out_col_major,
                         Allocator* alloc) {
    const bool is_3d = (fd > 0);
    if (fh <= 0 || fw <= 0 || (fh & 1) == 0 || (fw & 1) == 0 || (is_3d && (fd & 1) == 0)) {
        return ImageStatus::kInvalidArgument;
    }
    const usize d_eff = is_3d ? pages : 1;
    const usize plane = height * width;
    const usize total = plane * d_eff;
    if (src_col_major.size() < total || out_col_major.size() < total) {
        return ImageStatus::kInvalidArgument;
    }
    if (total == 0) {
        return ImageStatus::kOk;
    }

    const i32 h = as<i32>(height);
    const i32 w = as<i32>(width);
    const i32 d = as<i32>(d_eff);
    const i32 hh = fh / 2;
    const i32 hw = fw / 2;
    const i32 hd = is_3d ? (fd / 2) : 0;

    auto reflect = [&](i32 v, i32 n) noexcept -> i32 {
        if (pad_mode == PadMode::kConstant) {
            return (v < 0 || v >= n) ? -1 : v;
        }
        return fold_index(v, n, pad_mode);
    };

    ArenaTemp scratch = get_scratch(alloc);
    Array<f64> vals(scratch.allocator());
    vals.reserve(as<usize>(fh) * as<usize>(fw) * as<usize>(is_3d ? fd : 1));

    for (i32 p = 0; p < d; ++p) {
        for (i32 c = 0; c < w; ++c) {
            for (i32 r = 0; r < h; ++r) {
                vals.clear();
                for (i32 dp = -hd; dp <= hd; ++dp) {
                    const i32 ps = is_3d ? reflect(p + dp, d) : 0;
                    for (i32 dc = -hw; dc <= hw; ++dc) {
                        const i32 cs = reflect(c + dc, w);
                        for (i32 dr = -hh; dr <= hh; ++dr) {
                            const i32 rs = reflect(r + dr, h);
                            if (rs < 0 || cs < 0 || (is_3d && ps < 0)) {
                                if (pad_mode == PadMode::kConstant) {
                                    vals.push_back(0.0);
                                }
                            } else {
                                const f64 val = src_col_major[as<usize>(ps) * plane +
                                                              as<usize>(cs) * height +
                                                              as<usize>(rs)];
                                if (!is_nan(val)) {
                                    vals.push_back(val);
                                }
                            }
                        }
                    }
                }
                if (vals.empty()) {
                    out_col_major[as<usize>(p) * plane + as<usize>(c) * height + as<usize>(r)] =
                        kNaN;
                    continue;
                }
                sort(vals.as_mut_ref());
                i32 best_cnt = -1;
                f64 best_val = 0.0;
                for (usize vi = 0; vi < vals.size();) {
                    usize vj = vi + 1;
                    while (vj < vals.size() && vals[vj] == vals[vi]) {
                        ++vj;
                    }
                    const i32 cnt = as<i32>(vj - vi);
                    if (cnt > best_cnt) {
                        best_cnt = cnt;
                        best_val = vals[vi];
                    }
                    vi = vj;
                }
                out_col_major[as<usize>(p) * plane + as<usize>(c) * height + as<usize>(r)] =
                    best_val;
            }
        }
    }
    return ImageStatus::kOk;
}

ImageStatus medfilt2_u8(ArrayRef<u8> src_col_major, usize height, usize width, i32 rows, i32 cols,
                        MutArrayRef<u8> out_col_major, Allocator* alloc) {
    const usize n = height * width;
    if (rows <= 0 || cols <= 0 || src_col_major.size() < n || out_col_major.size() < n) {
        return ImageStatus::kInvalidArgument;
    }
    if (n == 0) {
        return ImageStatus::kOk;
    }

    ArenaTemp scratch = get_scratch(alloc);
    if ((rows & 1) != 0 && (cols & 1) != 0) {
        const i32 h = as<i32>(height);
        const i32 w = as<i32>(width);
        const i32 hr = (rows - 1) / 2;
        const i32 hc = (cols - 1) / 2;
        const i32 hp = h + rows - 1;
        const i32 wp = w + cols - 1;
        Array<u8> pp(as<usize>(hp) * as<usize>(wp), u8{0}, scratch.allocator());
        for (i32 c = 0; c < w; ++c) {
            const usize scol = as<usize>(c) * height;
            u8* pcol = pp.data() + as<usize>(c + hc) * as<usize>(hp) + as<usize>(hr);
            for (i32 r = 0; r < h; ++r) {
                pcol[r] = src_col_major[scol + as<usize>(r)];
            }
        }

        const i32 th = (rows * cols) / 2;
        i32 hist[256];
        for (i32 oc = 0; oc < w; ++oc) {
            for (i32 i = 0; i < 256; ++i) {
                hist[i] = 0;
            }
            for (i32 wc = 0; wc < cols; ++wc) {
                const u8* col = pp.data() + as<usize>(oc + wc) * as<usize>(hp);
                for (i32 wr = 0; wr < rows; ++wr) {
                    ++hist[col[wr]];
                }
            }
            i32 med = 0;
            i32 below = 0;
            while (below + hist[med] <= th) {
                below += hist[med];
                ++med;
            }
            out_col_major[as<usize>(oc) * height] = as<u8>(med);

            for (i32 orow = 1; orow < h; ++orow) {
                const i32 r_out = orow - 1;
                const i32 r_in = orow + rows - 1;
                for (i32 wc = 0; wc < cols; ++wc) {
                    const u8* col = pp.data() + as<usize>(oc + wc) * as<usize>(hp);
                    const u8 vo = col[r_out];
                    --hist[vo];
                    if (as<i32>(vo) < med) {
                        --below;
                    }
                    const u8 vi = col[r_in];
                    ++hist[vi];
                    if (as<i32>(vi) < med) {
                        ++below;
                    }
                }
                while (below > th) {
                    --med;
                    below -= hist[med];
                }
                while (below + hist[med] <= th) {
                    below += hist[med];
                    ++med;
                }
                out_col_major[as<usize>(oc) * height + as<usize>(orow)] = as<u8>(med);
            }
        }
        return ImageStatus::kOk;
    }

    Array<f64> src_f(n, scratch.allocator());
    Array<f64> out_f(n, scratch.allocator());
    for (usize i = 0; i < n; ++i) {
        src_f[i] = as<f64>(src_col_major[i]);
    }
    const ImageStatus st = medfilt2_f64(src_f, height, width, rows, cols, out_f, scratch.allocator());
    if (st != ImageStatus::kOk) {
        return st;
    }
    for (usize i = 0; i < n; ++i) {
        out_col_major[i] = as<u8>(round(clamp(out_f[i], 0.0, 255.0)));
    }
    return ImageStatus::kOk;
}

ImageStatus medfilt2_f64(ArrayRef<f64> src_col_major, usize height, usize width, i32 rows, i32 cols,
                         MutArrayRef<f64> out_col_major, Allocator* alloc) {
    const usize n = height * width;
    if (rows <= 0 || cols <= 0 || src_col_major.size() < n || out_col_major.size() < n) {
        return ImageStatus::kInvalidArgument;
    }
    if (n == 0) {
        return ImageStatus::kOk;
    }

    ArenaTemp scratch = get_scratch(alloc);
    Array<f64> src_copy(scratch.allocator());
    ArrayRef<f64> src = src_col_major;
    if (out_col_major.data() == src_col_major.data()) {
        src_copy.resize(n);
        for (usize i = 0; i < n; ++i) src_copy[i] = src_col_major[i];
        src = src_copy.as_ref();
    }

    const i32 h = as<i32>(height);
    const i32 w = as<i32>(width);
    if (rows == 3 && cols == 3) {
        for (i32 oc = 0; oc < w; ++oc) {
            for (i32 orow = 0; orow < h; ++orow) {
                f64 p[9];
                usize m = 0;
                for (i32 kj = 0; kj < 3; ++kj) {
                    for (i32 ki = 0; ki < 3; ++ki) {
                        const i32 r_in = orow + ki - 1;
                        const i32 c_in = oc + kj - 1;
                        p[m++] = (r_in < 0 || r_in >= h || c_in < 0 || c_in >= w)
                                     ? 0.0
                                     : src[as<usize>(c_in) * height + as<usize>(r_in)];
                    }
                }
                out_col_major[as<usize>(oc) * height + as<usize>(orow)] = med9(p);
            }
        }
        return ImageStatus::kOk;
    }

    const i32 half_r = rows / 2;
    const i32 half_c = cols / 2;
    Array<f64> window(as<usize>(rows) * as<usize>(cols), 0.0, scratch.allocator());

    for (i32 oc = 0; oc < w; ++oc) {
        for (i32 orow = 0; orow < h; ++orow) {
            usize idx = 0;
            for (i32 kj = 0; kj < cols; ++kj) {
                const i32 c_in = oc + kj - half_c;
                for (i32 ki = 0; ki < rows; ++ki) {
                    const i32 r_in = orow + ki - half_r;
                    window[idx++] =
                        (r_in < 0 || r_in >= h || c_in < 0 || c_in >= w)
                            ? 0.0
                            : src[as<usize>(c_in) * height + as<usize>(r_in)];
                }
            }
            out_col_major[as<usize>(oc) * height + as<usize>(orow)] =
                median_of_buffer(window.data(), idx);
        }
    }
    return ImageStatus::kOk;
}

ImageStatus imsharpen_2d(ArrayRef<f64> img_col_major, usize height, usize width, f64 radius,
                         f64 amount, f64 threshold, MutArrayRef<f64> out_col_major,
                         Allocator* alloc) {
    const usize n = height * width;
    if (img_col_major.size() < n || out_col_major.size() < n || !is_finite(amount) ||
        !is_finite(threshold) || !(threshold >= 0.0 && threshold <= 1.0) ||
        (radius > 0.0 && (!is_finite(radius) || radius > 1e6))) {
        return ImageStatus::kInvalidArgument;
    }
    if (n == 0) {
        return ImageStatus::kOk;
    }
    if (!(radius > 0.0)) {
        radius = 1.0;
    }
    i32 fs = 2 * as<i32>(ceil(2.0 * radius)) + 1;
    if (fs < 3) {
        fs = 3;
    }
    if ((fs & 1) == 0) {
        fs += 1;
    }

    ArenaTemp scratch = get_scratch(alloc);
    Array<f64> blurred(n, 0.0, scratch.allocator());
    const ImageStatus st =
        gaussian_filter_2d(img_col_major, height, width, radius, fs, false, blurred,
                           scratch.allocator());
    if (st != ImageStatus::kOk) {
        return st;
    }

    if (threshold == 0.0) {
        for (usize i = 0; i < n; ++i) {
            out_col_major[i] = img_col_major[i] + amount * (img_col_major[i] - blurred[i]);
        }
        return ImageStatus::kOk;
    }

    Array<f64> high(n, 0.0, scratch.allocator());
    f64 max_abs = 0.0;
    for (usize i = 0; i < n; ++i) {
        const f64 v = img_col_major[i] - blurred[i];
        high[i] = v;
        const f64 a = abs(v);
        if (a > max_abs) {
            max_abs = a;
        }
    }
    const f64 thr_abs = threshold * max_abs;
    for (usize i = 0; i < n; ++i) {
        const f64 hval = (abs(high[i]) < thr_abs) ? 0.0 : high[i];
        out_col_major[i] = img_col_major[i] + amount * hval;
    }
    return ImageStatus::kOk;
}

ImageStatus imbilatfilt_2d(ArrayRef<f64> img_col_major, usize height, usize width,
                           f64 degree_of_smoothing, f64 spatial_sigma,
                           MutArrayRef<f64> out_col_major, Allocator* alloc) {
    const usize n = height * width;
    if (img_col_major.size() < n || out_col_major.size() < n || !is_finite(degree_of_smoothing) ||
        !(degree_of_smoothing > 0.0) ||
        (spatial_sigma > 0.0 && (!is_finite(spatial_sigma) || spatial_sigma > 1e6))) {
        return ImageStatus::kInvalidArgument;
    }
    if (n == 0) {
        return ImageStatus::kOk;
    }
    if (!(spatial_sigma > 0.0)) {
        spatial_sigma = 1.0;
    }

    const i32 half = max(1, as<i32>(ceil(2.0 * spatial_sigma)));
    const i32 win = 2 * half + 1;
    const f64 inv2ss = 0.5 / (spatial_sigma * spatial_sigma);
    const f64 inv2dos = 0.5 / degree_of_smoothing;

    ArenaTemp scratch = get_scratch(alloc);
    Array<f64> sw(as<usize>(win) * as<usize>(win), 0.0, scratch.allocator());
    for (i32 di = -half; di <= half; ++di) {
        for (i32 dj = -half; dj <= half; ++dj) {
            sw[as<usize>(di + half) * as<usize>(win) + as<usize>(dj + half)] =
                exp(-as<f64>(di * di + dj * dj) * inv2ss);
        }
    }

    const i32 h = as<i32>(height);
    const i32 w = as<i32>(width);
    for (i32 oc = 0; oc < w; ++oc) {
        for (i32 orow = 0; orow < h; ++orow) {
            const f64 ic = img_col_major[as<usize>(oc) * height + as<usize>(orow)];
            f64 sum_w = 0.0;
            f64 sum_v = 0.0;
            for (i32 dj = -half; dj <= half; ++dj) {
                const i32 c = clamp(oc + dj, 0, w - 1);
                for (i32 di = -half; di <= half; ++di) {
                    const i32 r = clamp(orow + di, 0, h - 1);
                    const f64 v = img_col_major[as<usize>(c) * height + as<usize>(r)];
                    const f64 di_v = v - ic;
                    const f64 rw = exp(-(di_v * di_v) * inv2dos);
                    const f64 wt =
                        sw[as<usize>(di + half) * as<usize>(win) + as<usize>(dj + half)] * rw;
                    sum_w += wt;
                    sum_v += wt * v;
                }
            }
            out_col_major[as<usize>(oc) * height + as<usize>(orow)] =
                (sum_w > 0.0) ? (sum_v / sum_w) : ic;
        }
    }
    return ImageStatus::kOk;
}

ImageStatus stdfilt_2d(ArrayRef<f64> img_col_major, usize height, usize width,
                       ArrayRef<u8> domain_col_major, usize dom_h, usize dom_w,
                       MutArrayRef<f64> out_col_major, Allocator* alloc) {
    const usize n = height * width;
    if (img_col_major.size() < n || out_col_major.size() < n ||
        domain_col_major.size() < dom_h * dom_w) {
        return ImageStatus::kInvalidArgument;
    }
    if (n == 0) {
        return ImageStatus::kOk;
    }

    const i32 h = as<i32>(height);
    const i32 w = as<i32>(width);
    const i32 kh = as<i32>(dom_h);
    const i32 kw = as<i32>(dom_w);

    bool rect = (kh > 0 && kw > 0);
    for (usize i = 0; rect && i < dom_h * dom_w; ++i) {
        if (domain_col_major[i] == 0) {
            rect = false;
        }
    }

    ArenaTemp scratch = get_scratch(alloc);
    if (rect) {
        const i32 rh = kh / 2;
        const i32 ch = kw / 2;
        const f64 count = as<f64>(kh) * as<f64>(kw);
        const i32 hs = h + kh;
        const i32 ws = w + kw;
        Array<f64> sx(as<usize>(hs) * as<usize>(ws), 0.0, scratch.allocator());
        Array<f64> sxx(as<usize>(hs) * as<usize>(ws), 0.0, scratch.allocator());
        auto x_at = [&](i32 r, i32 c) noexcept -> f64& {
            return sx[as<usize>(c) * as<usize>(hs) + as<usize>(r)];
        };
        auto xx_at = [&](i32 r, i32 c) noexcept -> f64& {
            return sxx[as<usize>(c) * as<usize>(hs) + as<usize>(r)];
        };
        for (i32 c = 1; c < ws; ++c) {
            const i32 sc = fold_index((c - 1) - ch, w, PadMode::kSymmetric);
            const usize scol = as<usize>(sc) * height;
            for (i32 r = 1; r < hs; ++r) {
                const i32 sr = fold_index((r - 1) - rh, h, PadMode::kSymmetric);
                const f64 p = img_col_major[scol + as<usize>(sr)];
                x_at(r, c) = p + x_at(r - 1, c) + x_at(r, c - 1) - x_at(r - 1, c - 1);
                xx_at(r, c) = p * p + xx_at(r - 1, c) + xx_at(r, c - 1) - xx_at(r - 1, c - 1);
            }
        }
        for (i32 c = 0; c < w; ++c) {
            for (i32 r = 0; r < h; ++r) {
                const f64 s =
                    x_at(r + kh, c + kw) - x_at(r, c + kw) - x_at(r + kh, c) + x_at(r, c);
                const f64 s2 =
                    xx_at(r + kh, c + kw) - xx_at(r, c + kw) - xx_at(r + kh, c) + xx_at(r, c);
                f64 var = (count > 1.0) ? ((s2 - s * s / count) / (count - 1.0)) : 0.0;
                if (var < 0.0) {
                    var = 0.0;
                }
                out_col_major[as<usize>(c) * height + as<usize>(r)] = sqrt(var);
            }
        }
        return ImageStatus::kOk;
    }

    Array<Off2D> offs(scratch.allocator());
    build_domain_offsets(domain_col_major, dom_h, dom_w, offs);
    const usize m = offs.size();
    if (m == 0) {
        for (usize i = 0; i < n; ++i) {
            out_col_major[i] = 0.0;
        }
        return ImageStatus::kOk;
    }
    for (i32 c = 0; c < w; ++c) {
        for (i32 r = 0; r < h; ++r) {
            f64 sum = 0.0;
            for (usize k = 0; k < m; ++k) {
                sum += sample_sym_2d(img_col_major, r + offs[k].dr, c + offs[k].dc, h, w);
            }
            const f64 mu = sum / as<f64>(m);
            f64 v = 0.0;
            for (usize k = 0; k < m; ++k) {
                const f64 d = sample_sym_2d(img_col_major, r + offs[k].dr, c + offs[k].dc, h, w) - mu;
                v += d * d;
            }
            out_col_major[as<usize>(c) * height + as<usize>(r)] =
                (m > 1) ? sqrt(v / as<f64>(m - 1)) : 0.0;
        }
    }
    return ImageStatus::kOk;
}

ImageStatus ordfilt2_2d(ArrayRef<f64> img_col_major, usize height, usize width, i32 nth,
                        ArrayRef<u8> domain_col_major, usize dom_h, usize dom_w,
                        ArrayRef<f64> additive_offsets, PadMode boundary, f64 pad_value,
                        MutArrayRef<f64> out_col_major, Allocator* alloc) {
    const usize n = height * width;
    const usize dom_n = dom_h * dom_w;
    if (img_col_major.size() < n || out_col_major.size() < n || domain_col_major.size() < dom_n) {
        return ImageStatus::kInvalidArgument;
    }
    if (n == 0) {
        return ImageStatus::kOk;
    }

    const bool has_s = (additive_offsets.size() == dom_n);
    const i32 rh = as<i32>(dom_h) / 2;
    const i32 ch = as<i32>(dom_w) / 2;

    struct OffS {
        i32 dr;
        i32 dc;
        f64 s;
    };
    ArenaTemp scratch = get_scratch(alloc);
    Array<OffS> offs(scratch.allocator());
    offs.reserve(dom_n);
    for (usize c = 0; c < dom_w; ++c) {
        for (usize r = 0; r < dom_h; ++r) {
            const usize idx = c * dom_h + r;
            if (domain_col_major[idx] != 0) {
                const f64 sv = has_s ? additive_offsets[idx] : 0.0;
                offs.push_back(OffS{as<i32>(r) - rh, as<i32>(c) - ch, sv});
            }
        }
    }
    const i32 m = as<i32>(offs.size());
    if (m == 0) {
        for (usize i = 0; i < n; ++i) {
            out_col_major[i] = 0.0;
        }
        return ImageStatus::kOk;
    }
    if (nth < 1 || nth > m) {
        return ImageStatus::kInvalidArgument;
    }

    const i32 h = as<i32>(height);
    const i32 w = as<i32>(width);
    Array<f64> nb(as<usize>(m), 0.0, scratch.allocator());
    const usize target = as<usize>(nth - 1);

    for (i32 c = 0; c < w; ++c) {
        for (i32 r = 0; r < h; ++r) {
            for (i32 k = 0; k < m; ++k) {
                const OffS& o = offs[as<usize>(k)];
                nb[as<usize>(k)] =
                    sample_padded_2d(img_col_major, r + o.dr, c + o.dc, h, w, boundary, pad_value) +
                    o.s;
            }
            nth_select_f64(nb.data(), as<usize>(m), target);
            out_col_major[as<usize>(c) * height + as<usize>(r)] = nb[target];
        }
    }
    return ImageStatus::kOk;
}

ImageStatus entropyfilt_u8(ArrayRef<u8> img_col_major, usize height, usize width, i32 nbins,
                           ArrayRef<u8> domain_col_major, usize dom_h, usize dom_w,
                           MutArrayRef<f64> out_col_major, Allocator* alloc) {
    const usize n = height * width;
    if (nbins <= 0 || nbins > 256 || img_col_major.size() < n || out_col_major.size() < n ||
        domain_col_major.size() < dom_h * dom_w) {
        return ImageStatus::kInvalidArgument;
    }
    if (n == 0) {
        return ImageStatus::kOk;
    }

    ArenaTemp scratch = get_scratch(alloc);
    Array<Off2D> offs(scratch.allocator());
    build_domain_offsets(domain_col_major, dom_h, dom_w, offs);
    const usize m = offs.size();
    if (m == 0) {
        for (usize i = 0; i < n; ++i) {
            out_col_major[i] = 0.0;
        }
        return ImageStatus::kOk;
    }

    const i32 h = as<i32>(height);
    const i32 w = as<i32>(width);
    const f64 total = as<f64>(m);
    i32 hist[256];

    for (i32 c = 0; c < w; ++c) {
        for (i32 r = 0; r < h; ++r) {
            for (i32 b = 0; b < nbins; ++b) {
                hist[b] = 0;
            }
            for (usize k = 0; k < m; ++k) {
                const i32 rs = fold_index(r + offs[k].dr, h, PadMode::kSymmetric);
                const i32 cs = fold_index(c + offs[k].dc, w, PadMode::kSymmetric);
                i32 b = as<i32>(img_col_major[as<usize>(cs) * height + as<usize>(rs)]);
                if (b >= nbins) {
                    b = nbins - 1;
                }
                ++hist[b];
            }
            f64 h_bits = 0.0;
            for (i32 b = 0; b < nbins; ++b) {
                if (hist[b] <= 0) {
                    continue;
                }
                const f64 p = as<f64>(hist[b]) / total;
                h_bits -= p * log2(p);
            }
            out_col_major[as<usize>(c) * height + as<usize>(r)] = h_bits;
        }
    }
    return ImageStatus::kOk;
}

ImageStatus rangefilt_2d(ArrayRef<f64> img_col_major, usize height, usize width,
                         ArrayRef<u8> domain_col_major, usize dom_h, usize dom_w,
                         MutArrayRef<f64> out_col_major, Allocator* alloc) {
    const usize n = height * width;
    if (img_col_major.size() < n || out_col_major.size() < n ||
        domain_col_major.size() < dom_h * dom_w) {
        return ImageStatus::kInvalidArgument;
    }
    if (n == 0) {
        return ImageStatus::kOk;
    }

    ArenaTemp scratch = get_scratch(alloc);
    Array<Off2D> offs(scratch.allocator());
    build_domain_offsets(domain_col_major, dom_h, dom_w, offs);
    const usize m = offs.size();
    if (m == 0) {
        for (usize i = 0; i < n; ++i) {
            out_col_major[i] = 0.0;
        }
        return ImageStatus::kOk;
    }

    const i32 h = as<i32>(height);
    const i32 w = as<i32>(width);
    const f64 pos_inf = Limits<f64>::infinity();
    for (i32 c = 0; c < w; ++c) {
        for (i32 r = 0; r < h; ++r) {
            f64 mn = pos_inf;
            f64 mx = -pos_inf;
            for (usize k = 0; k < m; ++k) {
                const f64 v = sample_sym_2d(img_col_major, r + offs[k].dr, c + offs[k].dc, h, w);
                if (v < mn) {
                    mn = v;
                }
                if (v > mx) {
                    mx = v;
                }
            }
            out_col_major[as<usize>(c) * height + as<usize>(r)] = mx - mn;
        }
    }
    return ImageStatus::kOk;
}

ImageStatus wiener2_2d(ArrayRef<f64> unit_col_major, usize height, usize width, usize nh, usize nw,
                       f64 noise_in, MutArrayRef<f64> out_col_major, f64& noise_out,
                       Allocator* alloc) {
    const usize n = height * width;
    if (unit_col_major.size() < n || out_col_major.size() < n ||
        (!is_nan(noise_in) && (!is_finite(noise_in) || noise_in < 0.0))) {
        return ImageStatus::kInvalidArgument;
    }
    if (n == 0) {
        noise_out = 0.0;
        return ImageStatus::kOk;
    }
    if (nh == 0) {
        nh = 3;
    }
    if (nw == 0) {
        nw = 3;
    }

    // Zero-padded summed-area tables matching conv2(I, ones(nh,nw)/(nh*nw), "same"):
    // for output (r, c), window spans rows [r - (nh-1)/2, r + nh/2], cols [c - (nw-1)/2, c + nw/2].
    const usize hs = height + 1;
    const usize ws = width + 1;
    ArenaTemp scratch = get_scratch(alloc);
    Array<f64> sat1(hs * ws, 0.0, scratch.allocator());
    Array<f64> sat2(hs * ws, 0.0, scratch.allocator());
    auto s1 = [&](usize r, usize c) noexcept -> f64& { return sat1[c * hs + r]; };
    auto s2 = [&](usize r, usize c) noexcept -> f64& { return sat2[c * hs + r]; };

    for (usize c = 1; c < ws; ++c) {
        for (usize r = 1; r < hs; ++r) {
            const f64 v = unit_col_major[(c - 1) * height + (r - 1)];
            s1(r, c) = v + s1(r - 1, c) + s1(r, c - 1) - s1(r - 1, c - 1);
            s2(r, c) = v * v + s2(r - 1, c) + s2(r, c - 1) - s2(r - 1, c - 1);
        }
    }

    const i64 hi = as<i64>(height);
    const i64 wi = as<i64>(width);
    const i64 r_lo_off = as<i64>((nh - 1) / 2);
    const i64 r_hi_off = as<i64>(nh / 2);
    const i64 c_lo_off = as<i64>((nw - 1) / 2);
    const i64 c_hi_off = as<i64>(nw / 2);
    const f64 inv_area = 1.0 / as<f64>(nh * nw);

    Array<f64> mean_im(n, 0.0, scratch.allocator());
    Array<f64> var_im(n, 0.0, scratch.allocator());
    f64 var_sum = 0.0;

    for (i64 c = 0; c < wi; ++c) {
        const usize c0 = as<usize>(max<i64>(0, c - c_lo_off));
        const usize c1 = as<usize>(min<i64>(wi, c + c_hi_off + 1));
        for (i64 r = 0; r < hi; ++r) {
            const usize r0 = as<usize>(max<i64>(0, r - r_lo_off));
            const usize r1 = as<usize>(min<i64>(hi, r + r_hi_off + 1));
            const f64 sum1 = s1(r1, c1) - s1(r0, c1) - s1(r1, c0) + s1(r0, c0);
            const f64 sum2 = s2(r1, c1) - s2(r0, c1) - s2(r1, c0) + s2(r0, c0);
            const f64 mu = sum1 * inv_area;
            const f64 ms = sum2 * inv_area;
            const f64 vr = max(0.0, ms - mu * mu);
            const usize idx = as<usize>(c) * height + as<usize>(r);
            mean_im[idx] = mu;
            var_im[idx] = vr;
            var_sum += vr;
        }
    }

    f64 noise = noise_in;
    if (is_nan(noise)) {
        noise = var_sum / as<f64>(n);
    }
    noise_out = noise;

    for (usize i = 0; i < n; ++i) {
        f64 vo = var_im[i] - noise;
        if (vo < 0.0) {
            vo = 0.0;
        }
        const f64 denom = vo + noise;
        const f64 wt = (denom > 0.0) ? (vo / denom) : 0.0;
        out_col_major[i] = mean_im[i] + wt * (unit_col_major[i] - mean_im[i]);
    }
    return ImageStatus::kOk;
}

ImageStatus imboxfilt3_f64(ArrayRef<f64> vol_col_major, usize height, usize width, usize pages,
                           i32 fh, i32 fw, i32 fp, MutArrayRef<f64> out_col_major,
                           Allocator* alloc) {
    const usize total = height * width * pages;
    if (vol_col_major.size() < total || out_col_major.size() < total) {
        return ImageStatus::kInvalidArgument;
    }
    if (total == 0) {
        return ImageStatus::kOk;
    }
    if (fh <= 0) fh = 3;
    if (fw <= 0) fw = 3;
    if (fp <= 0) fp = 3;
    if ((fh & 1) == 0) fh += 1;
    if ((fw & 1) == 0) fw += 1;
    if ((fp & 1) == 0) fp += 1;

    const i32 h = as<i32>(height);
    const i32 w = as<i32>(width);
    const i32 p_cnt = as<i32>(pages);
    const i32 hh = fh / 2;
    const i32 hw = fw / 2;
    const i32 hp = fp / 2;
    const usize plane = height * width;

    ArenaTemp scratch = get_scratch(alloc);
    Array<f64> tmp1(total, 0.0, scratch.allocator());
    Array<f64> tmp2(total, 0.0, scratch.allocator());

    // Pass 1: H axis
    for (i32 p = 0; p < p_cnt; ++p) {
        for (i32 c = 0; c < w; ++c) {
            for (i32 r = 0; r < h; ++r) {
                f64 acc = 0.0;
                for (i32 k = -hh; k <= hh; ++k) {
                    const i32 rs = fold_index(r + k, h, PadMode::kReplicate);
                    acc += vol_col_major[as<usize>(p) * plane + as<usize>(c) * height +
                                         as<usize>(rs)];
                }
                tmp1[as<usize>(p) * plane + as<usize>(c) * height + as<usize>(r)] =
                    acc / as<f64>(fh);
            }
        }
    }
    // Pass 2: W axis
    for (i32 p = 0; p < p_cnt; ++p) {
        for (i32 r = 0; r < h; ++r) {
            for (i32 c = 0; c < w; ++c) {
                f64 acc = 0.0;
                for (i32 k = -hw; k <= hw; ++k) {
                    const i32 cs = fold_index(c + k, w, PadMode::kReplicate);
                    acc += tmp1[as<usize>(p) * plane + as<usize>(cs) * height + as<usize>(r)];
                }
                tmp2[as<usize>(p) * plane + as<usize>(c) * height + as<usize>(r)] =
                    acc / as<f64>(fw);
            }
        }
    }
    // Pass 3: P axis
    for (i32 p = 0; p < p_cnt; ++p) {
        for (i32 c = 0; c < w; ++c) {
            for (i32 r = 0; r < h; ++r) {
                f64 acc = 0.0;
                for (i32 k = -hp; k <= hp; ++k) {
                    const i32 ps = fold_index(p + k, p_cnt, PadMode::kReplicate);
                    acc += tmp2[as<usize>(ps) * plane + as<usize>(c) * height + as<usize>(r)];
                }
                out_col_major[as<usize>(p) * plane + as<usize>(c) * height + as<usize>(r)] =
                    acc / as<f64>(fp);
            }
        }
    }
    return ImageStatus::kOk;
}

ImageStatus imgaussfilt3_f64(ArrayRef<f64> vol_col_major, usize height, usize width, usize pages,
                             f64 sig_h, f64 sig_w, f64 sig_p, MutArrayRef<f64> out_col_major,
                             Allocator* alloc) {
    const usize total = height * width * pages;
    if (vol_col_major.size() < total || out_col_major.size() < total ||
        (sig_h > 0.0 && (!is_finite(sig_h) || sig_h > 1e6)) ||
        (sig_w > 0.0 && (!is_finite(sig_w) || sig_w > 1e6)) ||
        (sig_p > 0.0 && (!is_finite(sig_p) || sig_p > 1e6))) {
        return ImageStatus::kInvalidArgument;
    }
    if (total == 0) {
        return ImageStatus::kOk;
    }

    ArenaTemp scratch = get_scratch(alloc);
    const Array<f64> kh = gauss1d_kernel(sig_h, scratch.allocator());
    const Array<f64> kw = gauss1d_kernel(sig_w, scratch.allocator());
    const Array<f64> kp = gauss1d_kernel(sig_p, scratch.allocator());
    const i32 hh = as<i32>(kh.size()) / 2;
    const i32 hw = as<i32>(kw.size()) / 2;
    const i32 hp = as<i32>(kp.size()) / 2;

    const i32 h = as<i32>(height);
    const i32 w = as<i32>(width);
    const i32 p_cnt = as<i32>(pages);
    const usize plane = height * width;

    Array<f64> tmp1(total, 0.0, scratch.allocator());
    Array<f64> tmp2(total, 0.0, scratch.allocator());

    for (i32 p = 0; p < p_cnt; ++p) {
        for (i32 c = 0; c < w; ++c) {
            for (i32 r = 0; r < h; ++r) {
                f64 acc = 0.0;
                for (i32 k = -hh; k <= hh; ++k) {
                    const i32 rs = fold_index(r + k, h, PadMode::kReplicate);
                    acc += kh[as<usize>(k + hh)] *
                           vol_col_major[as<usize>(p) * plane + as<usize>(c) * height +
                                         as<usize>(rs)];
                }
                tmp1[as<usize>(p) * plane + as<usize>(c) * height + as<usize>(r)] = acc;
            }
        }
    }
    for (i32 p = 0; p < p_cnt; ++p) {
        for (i32 r = 0; r < h; ++r) {
            for (i32 c = 0; c < w; ++c) {
                f64 acc = 0.0;
                for (i32 k = -hw; k <= hw; ++k) {
                    const i32 cs = fold_index(c + k, w, PadMode::kReplicate);
                    acc += kw[as<usize>(k + hw)] *
                           tmp1[as<usize>(p) * plane + as<usize>(cs) * height + as<usize>(r)];
                }
                tmp2[as<usize>(p) * plane + as<usize>(c) * height + as<usize>(r)] = acc;
            }
        }
    }
    for (i32 p = 0; p < p_cnt; ++p) {
        for (i32 c = 0; c < w; ++c) {
            for (i32 r = 0; r < h; ++r) {
                f64 acc = 0.0;
                for (i32 k = -hp; k <= hp; ++k) {
                    const i32 ps = fold_index(p + k, p_cnt, PadMode::kReplicate);
                    acc += kp[as<usize>(k + hp)] *
                           tmp2[as<usize>(ps) * plane + as<usize>(c) * height + as<usize>(r)];
                }
                out_col_major[as<usize>(p) * plane + as<usize>(c) * height + as<usize>(r)] = acc;
            }
        }
    }
    return ImageStatus::kOk;
}

ImageStatus medfilt3_f64(ArrayRef<f64> vol_col_major, usize height, usize width, usize pages, i32 m,
                         i32 n, i32 p, MutArrayRef<f64> out_col_major, Allocator* alloc) {
    if (m <= 0) m = 3;
    if (n <= 0) n = 3;
    if (p <= 0) p = 3;
    if ((m & 1) == 0 || (n & 1) == 0 || (p & 1) == 0) {
        return ImageStatus::kInvalidArgument;
    }
    const usize plane = height * width;
    const usize total = plane * pages;
    if (vol_col_major.size() < total || out_col_major.size() < total) {
        return ImageStatus::kInvalidArgument;
    }
    if (total == 0) {
        return ImageStatus::kOk;
    }

    const i32 h = as<i32>(height);
    const i32 w = as<i32>(width);
    const i32 d = as<i32>(pages);
    const i32 hm = m / 2;
    const i32 hn = n / 2;
    const i32 hp = p / 2;
    const usize nbhd = as<usize>(m) * as<usize>(n) * as<usize>(p);

    ArenaTemp scratch = get_scratch(alloc);
    Array<f64> src_copy(scratch.allocator());
    ArrayRef<f64> src = vol_col_major;
    if (out_col_major.data() == vol_col_major.data()) {
        src_copy.resize(total);
        for (usize i = 0; i < total; ++i) src_copy[i] = vol_col_major[i];
        src = src_copy.as_ref();
    }
    Array<f64> window(nbhd, 0.0, scratch.allocator());

    for (i32 pi = 0; pi < d; ++pi) {
        for (i32 ci = 0; ci < w; ++ci) {
            for (i32 ri = 0; ri < h; ++ri) {
                usize idx = 0;
                for (i32 kp = -hp; kp <= hp; ++kp) {
                    const i32 ps = fold_index(pi + kp, d, PadMode::kSymmetric);
                    for (i32 kc = -hn; kc <= hn; ++kc) {
                        const i32 cs = fold_index(ci + kc, w, PadMode::kSymmetric);
                        for (i32 kr = -hm; kr <= hm; ++kr) {
                            const i32 rs = fold_index(ri + kr, h, PadMode::kSymmetric);
                            window[idx++] =
                                src[as<usize>(ps) * plane + as<usize>(cs) * height + as<usize>(rs)];
                        }
                    }
                }
                out_col_major[as<usize>(pi) * plane + as<usize>(ci) * height + as<usize>(ri)] =
                    median_of_buffer(window.data(), idx);
            }
        }
    }
    return ImageStatus::kOk;
}

ImageStatus convmtx2_f64(ArrayRef<f64> h_col_major, usize h_rows, usize h_cols, usize m, usize n,
                         MutArrayRef<f64> out_col_major) noexcept {
    if (h_rows == 0 || h_cols == 0 || m == 0 || n == 0 || h_col_major.size() < h_rows * h_cols) {
        return ImageStatus::kInvalidArgument;
    }
    const usize out_rows = m + h_rows - 1;
    const usize out_cols = n + h_cols - 1;
    const usize p_rows = out_rows * out_cols;
    const usize q_cols = m * n;
    if (out_col_major.size() < p_rows * q_cols) {
        return ImageStatus::kInvalidArgument;
    }
    for (usize i = 0; i < p_rows * q_cols; ++i) {
        out_col_major[i] = 0.0;
    }
    if (h_rows == 0 || h_cols == 0) {
        return ImageStatus::kOk;
    }

    for (usize c_in = 0; c_in < n; ++c_in) {
        for (usize r_in = 0; r_in < m; ++r_in) {
            const usize k = c_in * m + r_in;
            for (usize j = 0; j < h_cols; ++j) {
                for (usize i = 0; i < h_rows; ++i) {
                    const f64 hv = h_col_major[j * h_rows + i];
                    if (hv == 0.0) {
                        continue;
                    }
                    const usize out_r = r_in + i;
                    const usize out_c = c_in + j;
                    const usize p_idx = out_c * out_rows + out_r;
                    out_col_major[k * p_rows + p_idx] = hv;
                }
            }
        }
    }
    return ImageStatus::kOk;
}

ImageStatus freqz2_2d(ArrayRef<f64> h_col_major, usize h_rows, usize h_cols, usize m, usize n,
                      MutArrayRef<c64> out_h_col_major, MutArrayRef<f64> out_f1,
                      MutArrayRef<f64> out_f2, Allocator* alloc) {
    if (m == 0 || n == 0 || h_col_major.size() < h_rows * h_cols ||
        out_h_col_major.size() < m * n || out_f1.size() < m || out_f2.size() < n) {
        return ImageStatus::kInvalidArgument;
    }
    for (usize i = 0; i < m; ++i) {
        out_f1[i] = -1.0 + 2.0 * as<f64>(i) / as<f64>(m);
    }
    for (usize j = 0; j < n; ++j) {
        out_f2[j] = -1.0 + 2.0 * as<f64>(j) / as<f64>(n);
    }
    if (h_rows == 0 || h_cols == 0) {
        for (usize i = 0; i < m * n; ++i) {
            out_h_col_major[i] = c64(0.0, 0.0);
        }
        return ImageStatus::kOk;
    }

    const f64 cp = as<f64>((h_rows - 1) / 2);
    const f64 cq = as<f64>((h_cols - 1) / 2);
    ArenaTemp scratch = get_scratch(alloc);
    Array<c64> phase_p_i(m * h_rows, c64(0.0, 0.0), scratch.allocator());
    Array<c64> phase_q_j(n * h_cols, c64(0.0, 0.0), scratch.allocator());

    for (usize i = 0; i < m; ++i) {
        for (usize p = 0; p < h_rows; ++p) {
            const f64 a = kPi * out_f1[i] * (as<f64>(p) - cp);
            phase_p_i[p + i * h_rows] = c64(cos(a), sin(a));
        }
    }
    for (usize j = 0; j < n; ++j) {
        for (usize q = 0; q < h_cols; ++q) {
            const f64 a = kPi * out_f2[j] * (as<f64>(q) - cq);
            phase_q_j[q + j * h_cols] = c64(cos(a), sin(a));
        }
    }

    for (usize j = 0; j < n; ++j) {
        for (usize i = 0; i < m; ++i) {
            c64 sum(0.0, 0.0);
            for (usize q = 0; q < h_cols; ++q) {
                const c64 pq = phase_q_j[q + j * h_cols];
                for (usize p = 0; p < h_rows; ++p) {
                    sum += h_col_major[p + q * h_rows] * (phase_p_i[p + i * h_rows] * pq);
                }
            }
            out_h_col_major[i + j * m] = sum;
        }
    }
    return ImageStatus::kOk;
}

ImageStatus imdiffusefilt_2d(ArrayRef<f64> img_col_major, usize height, usize width,
                             ArrayRef<f64> grad_thresh_per_iter, bool maximal_conn,
                             bool exponential_cond, MutArrayRef<f64> out_col_major,
                             Allocator* alloc) {
    const usize n = height * width;
    if (img_col_major.size() < n || out_col_major.size() < n) {
        return ImageStatus::kInvalidArgument;
    }
    for (usize i = 0; i < grad_thresh_per_iter.size(); ++i) {
        if (!(grad_thresh_per_iter[i] > 0.0) || !is_finite(grad_thresh_per_iter[i])) {
            return ImageStatus::kInvalidArgument;
        }
    }
    if (n == 0) {
        return ImageStatus::kOk;
    }
    if (grad_thresh_per_iter.empty()) {
        for (usize i = 0; i < n; ++i) {
            out_col_major[i] = img_col_major[i];
        }
        return ImageStatus::kOk;
    }

    ArenaTemp scratch = get_scratch(alloc);
    Array<f64> img(ArrayRef<f64>(img_col_major.data(), n), scratch.allocator());
    Array<f64> upd(n, 0.0, scratch.allocator());
    const usize mp = height + 2;
    const usize wp = width + 2;
    Array<f64> pad(mp * wp, 0.0, scratch.allocator());

    auto conductance = [exponential_cond](f64 diff, f64 k_val) noexcept -> f64 {
        const f64 r = diff / k_val;
        if (exponential_cond) {
            return exp(-(r * r));
        }
        return 1.0 / (1.0 + r * r);
    };

    const f64 rate = maximal_conn ? (1.0 / 8.0) : (1.0 / 4.0);
    constexpr f64 kInvDiag2 = 0.5;

    for (usize it = 0; it < grad_thresh_per_iter.size(); ++it) {
        const f64 k_val = grad_thresh_per_iter[it];

        // Refresh replicate-padded buffer.
        for (usize c = 0; c < width; ++c) {
            for (usize r = 0; r < height; ++r) {
                pad[(c + 1) * mp + (r + 1)] = img[c * height + r];
            }
            pad[(c + 1) * mp + 0] = img[c * height + 0];
            pad[(c + 1) * mp + mp - 1] = img[c * height + height - 1];
        }
        for (usize r = 0; r < mp; ++r) {
            pad[0 * mp + r] = pad[1 * mp + r];
            pad[(wp - 1) * mp + r] = pad[width * mp + r];
        }

        auto p_at = [&](usize r, usize c) noexcept -> f64 { return pad[c * mp + r]; };

        for (usize c = 0; c < width; ++c) {
            for (usize r = 0; r < height; ++r) {
                const f64 ci = img[c * height + r];
                const f64 dn = p_at(r, c + 1) - ci;
                const f64 ds = p_at(r + 2, c + 1) - ci;
                const f64 de = p_at(r + 1, c + 2) - ci;
                const f64 dw = p_at(r + 1, c) - ci;
                f64 delta = conductance(dn, k_val) * dn + conductance(ds, k_val) * ds +
                            conductance(de, k_val) * de + conductance(dw, k_val) * dw;
                if (maximal_conn) {
                    const f64 dnw = p_at(r, c) - ci;
                    const f64 dne = p_at(r, c + 2) - ci;
                    const f64 dsw = p_at(r + 2, c) - ci;
                    const f64 dse = p_at(r + 2, c + 2) - ci;
                    delta += kInvDiag2 * (conductance(dnw, k_val) * dnw +
                                          conductance(dne, k_val) * dne +
                                          conductance(dsw, k_val) * dsw +
                                          conductance(dse, k_val) * dse);
                }
                upd[c * height + r] = ci + rate * delta;
            }
        }
        swap(img, upd);
    }

    for (usize i = 0; i < n; ++i) {
        out_col_major[i] = img[i];
    }
    return ImageStatus::kOk;
}

ImageStatus imgaborfilt_2d(ArrayRef<f64> img_col_major, usize height, usize width, f64 wavelength,
                           f64 orientation_deg, f64 sfb, f64 aspect,
                           MutArrayRef<f64> out_mag_col_major,
                           MutArrayRef<f64> out_phase_col_major, Allocator* alloc) {
    const usize n = height * width;
    if (!is_finite(wavelength) || !(wavelength >= 2.0) || !is_finite(orientation_deg) ||
        !is_finite(sfb) || !(sfb > 0.0) || !is_finite(aspect) || !(aspect > 0.0) ||
        img_col_major.size() < n || out_mag_col_major.size() < n ||
        out_phase_col_major.size() < n) {
        return ImageStatus::kInvalidArgument;
    }
    if (n == 0) {
        return ImageStatus::kOk;
    }

    const f64 ratio = (sfb > 60.0) ? 1.0 : ((pow(2.0, sfb) + 1.0) / (pow(2.0, sfb) - 1.0));
    const f64 sigma_x = wavelength / kPi * sqrt(log(2.0) * 0.5) * ratio;
    const f64 sigma_y = sigma_x / aspect;
    if (!is_finite(sigma_x) || !is_finite(sigma_y) || sigma_x > 1e5 || sigma_y > 1e5) {
        return ImageStatus::kInvalidArgument;
    }
    const i32 rx = as<i32>(ceil(7.0 * sigma_x));
    const i32 ry = as<i32>(ceil(7.0 * sigma_y));
    const i32 r = max(rx, ry);
    const usize ru = as<usize>(r);
    const usize mp = height + 2 * ru;
    const usize np = width + 2 * ru;
    const usize np_pix = mp * np;

    ArenaTemp scratch = get_scratch(alloc);
    Array<f64> apad(np_pix, 0.0, scratch.allocator());
    (void)padarray_f64(img_col_major, height, width, 1, r, r, r, r, 0, 0, PadMode::kReplicate, 0.0,
                       apad);

    Array<c64> buf0(np_pix, c64(0.0, 0.0), scratch.allocator());
    Array<c64> buf1(np_pix, c64(0.0, 0.0), scratch.allocator());
    for (usize i = 0; i < np_pix; ++i) {
        buf0[i] = c64(apad[i], 0.0);
    }

    const usize dims[2] = {mp, np};
    const fft::FftBatchLayout lay0 =
        fft::FftBatchLayout::from_col_major(ArrayRef<usize>(dims, 2), 0);
    const fft::FftBatchLayout lay1 =
        fft::FftBatchLayout::from_col_major(ArrayRef<usize>(dims, 2), 1);
    fft::fft_batch(buf0.as_ref(), buf1.as_mut_ref(), lay0);
    fft::fft_batch(buf1.as_ref(), buf0.as_mut_ref(), lay1);

    auto build_freq = [&](usize nlen, Array<f64>& out) {
        out.resize(nlen);
        if (nlen == 1) {
            out[0] = 0.0;
            return;
        }
        const f64 nl = as<f64>(nlen);
        const f64 lo = ((nlen & 1) != 0) ? (-0.5 + 1.0 / (2.0 * nl)) : -0.5;
        const f64 hi = ((nlen & 1) != 0) ? (0.5 - 1.0 / (2.0 * nl)) : (0.5 - 1.0 / nl);
        const f64 step = (hi - lo) / as<f64>(nlen - 1);
        for (usize i = 0; i < nlen; ++i) {
            out[i] = lo + step * as<f64>(i);
        }
    };
    Array<f64> u(scratch.allocator());
    Array<f64> v(scratch.allocator());
    build_freq(np, u);
    build_freq(mp, v);

    const f64 theta_rad = orientation_deg * kPi / 180.0;
    const f64 cos_t = cos(theta_rad);
    const f64 sin_t = sin(theta_rad);
    const f64 sigma_u = 1.0 / (2.0 * kPi * sigma_x);
    const f64 sigma_v = 1.0 / (2.0 * kPi * sigma_y);
    const f64 freq = 1.0 / wavelength;
    const f64 a_const = 2.0 * kPi * sigma_x * sigma_y;

    Array<f64> h_mask(np_pix, 0.0, scratch.allocator());
    for (usize c = 0; c < np; ++c) {
        const f64 uc = u[c];
        for (usize r2 = 0; r2 < mp; ++r2) {
            const f64 vc = v[r2];
            const f64 upr = uc * cos_t - vc * sin_t;
            const f64 vpr = uc * sin_t + vc * cos_t;
            const f64 e = -0.5 * ((upr - freq) * (upr - freq) / (sigma_u * sigma_u) +
                                  vpr * vpr / (sigma_v * sigma_v));
            h_mask[c * mp + r2] = a_const * exp(e);
        }
    }

    const usize ar = mp - (mp / 2);
    const usize ac = np - (np / 2);
    for (usize c = 0; c < np; ++c) {
        const usize cs = (c + ac) % np;
        for (usize r2 = 0; r2 < mp; ++r2) {
            const usize rs = (r2 + ar) % mp;
            const f64 hs = h_mask[cs * mp + rs];
            const usize idx = c * mp + r2;
            buf0[idx] = c64(buf0[idx].real() * hs, buf0[idx].imag() * hs);
        }
    }

    fft::ifft_batch(buf0.as_ref(), buf1.as_mut_ref(), lay0);
    fft::ifft_batch(buf1.as_ref(), buf0.as_mut_ref(), lay1);

    for (usize c = 0; c < width; ++c) {
        for (usize rr = 0; rr < height; ++rr) {
            const c64 z = buf0[(c + ru) * mp + (rr + ru)];
            const usize dst = c * height + rr;
            out_mag_col_major[dst] = hypot(z.real(), z.imag());
            out_phase_col_major[dst] = atan2(z.imag(), z.real());
        }
    }
    return ImageStatus::kOk;
}

f64 immerkaer_dos_2d(ArrayRef<f64> img_col_major, usize height, usize width) noexcept {
    if (height < 3 || width < 3 || img_col_major.size() < height * width) {
        return 1.0;
    }
    static constexpr f32 kLap[3][3] = {
        {1.0f, -2.0f, 1.0f}, {-2.0f, 4.0f, -2.0f}, {1.0f, -2.0f, 1.0f}};
    const usize mp = height + 2;
    const usize wp = width + 2;
    f32 sum_abs = 0.0f;
    for (usize oc = 0; oc < wp; ++oc) {
        for (usize orr = 0; orr < mp; ++orr) {
            f32 s = 0.0f;
            for (i32 i = 0; i < 3; ++i) {
                const i64 ir = as<i64>(orr) - as<i64>(i);
                if (ir < 0 || as<usize>(ir) >= height) {
                    continue;
                }
                for (i32 j = 0; j < 3; ++j) {
                    const i64 ic = as<i64>(oc) - as<i64>(j);
                    if (ic < 0 || as<usize>(ic) >= width) {
                        continue;
                    }
                    s += kLap[i][j] *
                         as<f32>(img_col_major[as<usize>(ic) * height + as<usize>(ir)]);
                }
            }
            sum_abs += abs(s);
        }
    }
    const f32 dos = sum_abs * sqrt(as<f32>(kPi) * 0.5f) /
                    (6.0f * as<f32>(width - 2) * as<f32>(height - 2));
    return (dos == 0.0f) ? as<f64>(Limits<f32>::kEpsilon) : as<f64>(dos);
}

ImageStatus imnlmfilt_2d(ArrayRef<f64> img_col_major, usize height, usize width, f64 dos,
                         i32 swsize, i32 cwsize, MutArrayRef<f64> out_col_major, f64& est_dos_out,
                         Allocator* alloc) {
    const usize n = height * width;
    if (height < 21 || width < 21 || swsize < 1 || (swsize & 1) == 0 || cwsize < 1 ||
        (cwsize & 1) == 0 || cwsize > swsize || as<usize>(swsize) > min(height, width) ||
        is_nan(dos) || (dos >= 0.0 && (!is_finite(dos) || !(dos > 0.0))) ||
        img_col_major.size() < n || out_col_major.size() < n) {
        return ImageStatus::kInvalidArgument;
    }

    const f64 h_val = (dos < 0.0) ? immerkaer_dos_2d(img_col_major, height, width) : dos;
    est_dos_out = h_val;

    const i32 sh = (swsize - 1) / 2;
    const i32 ch = (cwsize - 1) / 2;
    const i32 pad = sh + ch;
    const usize mp = height + as<usize>(2 * pad);
    const usize wp = width + as<usize>(2 * pad);

    ArenaTemp scratch = get_scratch(alloc);
    Array<f64> padded(mp * wp, 0.0, scratch.allocator());
    (void)padarray_f64(img_col_major, height, width, 1, pad, pad, pad, pad, 0, 0,
                       PadMode::kReplicate, 0.0, padded);

    const f64 inv_h2 = 1.0 / (h_val * h_val);
    const f64 inv_cwsq = 1.0 / as<f64>(cwsize * cwsize);
    const usize pad_u = as<usize>(pad);

    for (usize c = 0; c < width; ++c) {
        for (usize r = 0; r < height; ++r) {
            const usize pr = r + pad_u;
            const usize pc = c + pad_u;
            f64 sum_w = 0.0;
            f64 sum_wi = 0.0;
            f64 max_w = 0.0;

            for (i32 dc = -sh; dc <= sh; ++dc) {
                for (i32 dr = -sh; dr <= sh; ++dr) {
                    if (dr == 0 && dc == 0) {
                        continue;
                    }
                    const usize qr = as<usize>(as<i64>(pr) + as<i64>(dr));
                    const usize qc = as<usize>(as<i64>(pc) + as<i64>(dc));
                    f64 d2 = 0.0;
                    for (i32 ac = -ch; ac <= ch; ++ac) {
                        const usize p_col = as<usize>(as<i64>(pc) + as<i64>(ac)) * mp;
                        const usize q_col = as<usize>(as<i64>(qc) + as<i64>(ac)) * mp;
                        for (i32 ar = -ch; ar <= ch; ++ar) {
                            const f64 a = padded[p_col + as<usize>(as<i64>(pr) + as<i64>(ar))];
                            const f64 b = padded[q_col + as<usize>(as<i64>(qr) + as<i64>(ar))];
                            const f64 diff = a - b;
                            d2 += diff * diff;
                        }
                    }
                    d2 *= inv_cwsq;
                    const f64 wt = exp(-d2 * inv_h2);
                    sum_w += wt;
                    sum_wi += wt * padded[qc * mp + qr];
                    if (wt > max_w) {
                        max_w = wt;
                    }
                }
            }
            sum_w += max_w;
            sum_wi += max_w * img_col_major[c * height + r];
            out_col_major[c * height + r] =
                (sum_w > 0.0) ? (sum_wi / sum_w) : img_col_major[c * height + r];
        }
    }
    return ImageStatus::kOk;
}

} // namespace keel::image
