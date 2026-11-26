#include "keel/image.h"

#include "keel/algo.h"
#include "keel/fft.h"
#include "keel/math.h"

namespace keel::image {

namespace {

constexpr f64 kModSheppLogan[10][6] = {
    { 1.0,  0.69,   0.92,   0.0,    0.0,      0.0},
    {-0.8,  0.6624, 0.874,  0.0,   -0.0184,   0.0},
    {-0.2,  0.11,   0.31,   0.22,   0.0,    -18.0},
    {-0.2,  0.16,   0.41,  -0.22,   0.0,     18.0},
    { 0.1,  0.21,   0.25,   0.0,    0.35,     0.0},
    { 0.1,  0.046,  0.046,  0.0,    0.1,      0.0},
    { 0.1,  0.046,  0.046,  0.0,   -0.1,      0.0},
    { 0.1,  0.046,  0.023, -0.08,  -0.605,    0.0},
    { 0.1,  0.023,  0.023,  0.0,   -0.606,    0.0},
    { 0.1,  0.023,  0.046,  0.06,  -0.605,    0.0},
};

constexpr f64 kSheppLogan[10][6] = {
    { 1.0,   0.69,   0.92,   0.0,    0.0,       0.0},
    {-0.98,  0.6624, 0.874,  0.0,   -0.0184,    0.0},
    {-0.02,  0.11,   0.31,   0.22,   0.0,     -18.0},
    {-0.02,  0.16,   0.41,  -0.22,   0.0,      18.0},
    { 0.01,  0.21,   0.25,   0.0,    0.35,      0.0},
    { 0.01,  0.046,  0.046,  0.0,    0.1,       0.0},
    { 0.01,  0.046,  0.046,  0.0,   -0.1,       0.0},
    { 0.01,  0.046,  0.023, -0.08,  -0.605,     0.0},
    { 0.01,  0.023,  0.023,  0.0,   -0.606,     0.0},
    { 0.01,  0.023,  0.046,  0.06,  -0.605,     0.0},
};

void dct1d_makhoul(ArrayRef<f64> in, MutArrayRef<f64> out, MutArrayRef<f64> vbuf,
                   MutArrayRef<c64> fbuf, ArrayRef<c64> twiddles) noexcept {
    const usize n = in.size();
    if (n == 0) {
        return;
    }
    if (n == 1) {
        out[0] = in[0];
        return;
    }
    const usize half_even = (n + 1) / 2;
    const usize half_odd = n / 2;
    for (usize k = 0; k < half_even; ++k) {
        vbuf[k] = in[2 * k];
    }
    for (usize k = 0; k < half_odd; ++k) {
        vbuf[n - 1 - k] = in[2 * k + 1];
    }
    keel::fft::fft_1d_real(ArrayRef<f64>(vbuf.data(), n), MutArrayRef<c64>(fbuf.data(), n));
    for (usize k = 0; k < n; ++k) {
        const c64 prod = fbuf[k] * twiddles[k];
        out[k] = prod.re;
    }
}

void idct1d_makhoul(ArrayRef<f64> in, MutArrayRef<f64> out, MutArrayRef<c64> zbuf,
                    MutArrayRef<c64> ibuf, ArrayRef<c64> twiddles) noexcept {
    const usize n = in.size();
    if (n == 0) {
        return;
    }
    if (n == 1) {
        out[0] = in[0];
        return;
    }
    for (usize k = 0; k < n; ++k) {
        zbuf[k] = c64{in[k] * twiddles[k].re, in[k] * twiddles[k].im};
    }
    keel::fft::ifft_1d(ArrayRef<c64>(zbuf.data(), n), MutArrayRef<c64>(ibuf.data(), n));
    const usize half_even = (n + 1) / 2;
    const usize half_odd = n / 2;
    for (usize k = 0; k < half_even; ++k) {
        out[2 * k] = ibuf[k].re;
    }
    for (usize k = 0; k < half_odd; ++k) {
        out[2 * k + 1] = ibuf[n - 1 - k].re;
    }
}

void fill_dct_twiddles(usize n, bool inverse, MutArrayRef<c64> tw) noexcept {
    if (n == 0) {
        return;
    }
    const f64 nd = as<f64>(n);
    const f64 w0 = inverse ? sqrt(nd) : sqrt(1.0 / nd);
    const f64 wk = inverse ? sqrt(2.0 * nd) : sqrt(2.0 / nd);
    const f64 sign = inverse ? 1.0 : -1.0;
    const f64 step = sign * kPi / (2.0 * nd);
    tw[0] = c64{w0, 0.0};
    for (usize k = 1; k < n; ++k) {
        const f64 ang = step * as<f64>(k);
        tw[k] = c64{wk * cos(ang), wk * sin(ang)};
    }
}

usize next_pow2(usize n) noexcept {
    usize p = 1;
    while (p < n) {
        p <<= 1;
    }
    return p;
}

usize wrap_shift(usize idx, i64 shift, usize dim) noexcept {
    if (dim == 0) {
        return 0;
    }
    const i64 d = as<i64>(dim);
    i64 m = (as<i64>(idx) + shift) % d;
    if (m < 0) {
        m += d;
    }
    return as<usize>(m);
}

void fft2_col_major(ArrayRef<c64> src, MutArrayRef<c64> dst, usize h, usize w,
                    keel::fft::FftDirection dir, Array<c64>& tmp) {
    if (h == 1 && w == 1) {
        dst[0] = src[0];
        return;
    }
    const usize dims[2] = {h, w};
    if (h == 1) {
        const auto layout = keel::fft::FftBatchLayout::contiguous(1, w);
        keel::fft::fft_batch(src, dst, layout, dir);
        return;
    }
    if (w == 1) {
        const auto layout = keel::fft::FftBatchLayout::contiguous(1, h);
        keel::fft::fft_batch(src, dst, layout, dir);
        return;
    }
    const auto col_layout = keel::fft::FftBatchLayout::from_col_major(ArrayRef<usize>(dims, 2), 0);
    const auto row_layout = keel::fft::FftBatchLayout::from_col_major(ArrayRef<usize>(dims, 2), 1);
    keel::fft::fft_batch(src, tmp.as_mut_ref(), col_layout, dir);
    keel::fft::fft_batch(tmp.as_ref(), dst, row_layout, dir);
}

} // namespace

ImageStatus dct_matrix(usize n, MutArrayRef<f64> out_col_major) noexcept {
    if (out_col_major.size() != n * n) {
        return ImageStatus::kInvalidArgument;
    }
    if (n == 0) {
        return ImageStatus::kOk;
    }
    const f64 nd = as<f64>(n);
    const f64 w0 = sqrt(1.0 / nd);
    const f64 wk = sqrt(2.0 / nd);
    const f64 pi_over_2n = kPi / (2.0 * nd);
    for (usize col = 0; col < n; ++col) {
        const f64 two_n_plus_1 = as<f64>(2 * col + 1);
        for (usize k = 0; k < n; ++k) {
            const f64 w = (k == 0) ? w0 : wk;
            out_col_major[col * n + k] = w * cos(pi_over_2n * as<f64>(k) * two_n_plus_1);
        }
    }
    return ImageStatus::kOk;
}

ImageStatus dct_2d(ArrayRef<f64> in_col_major, usize height, usize width,
                   MutArrayRef<f64> out_col_major, Allocator* alloc) {
    const usize total = height * width;
    if (in_col_major.size() != total || out_col_major.size() != total) {
        return ImageStatus::kInvalidArgument;
    }
    if (total == 0) {
        return ImageStatus::kOk;
    }
    Allocator* a = alloc ? alloc : default_allocator();
    const usize max_dim = max(height, width);
    Array<f64> tmp(a);
    tmp.resize(total);
    Array<f64> line_in(a);
    line_in.resize(max_dim);
    Array<f64> line_out(a);
    line_out.resize(max_dim);
    Array<f64> vbuf(a);
    vbuf.resize(max_dim);
    Array<c64> fbuf(a);
    fbuf.resize(max_dim);
    Array<c64> tw(a);
    tw.resize(max_dim);

    // Pass 1: columns (length = height).
    fill_dct_twiddles(height, false, MutArrayRef<c64>(tw.data(), height));
    for (usize c = 0; c < width; ++c) {
        ArrayRef<f64> col_in(in_col_major.data() + c * height, height);
        MutArrayRef<f64> col_out(tmp.data() + c * height, height);
        dct1d_makhoul(col_in, col_out, MutArrayRef<f64>(vbuf.data(), height),
                      MutArrayRef<c64>(fbuf.data(), height), ArrayRef<c64>(tw.data(), height));
    }

    // Pass 2: rows (length = width).
    fill_dct_twiddles(width, false, MutArrayRef<c64>(tw.data(), width));
    for (usize r = 0; r < height; ++r) {
        for (usize c = 0; c < width; ++c) {
            line_in[c] = tmp[c * height + r];
        }
        dct1d_makhoul(ArrayRef<f64>(line_in.data(), width), MutArrayRef<f64>(line_out.data(), width),
                      MutArrayRef<f64>(vbuf.data(), width), MutArrayRef<c64>(fbuf.data(), width),
                      ArrayRef<c64>(tw.data(), width));
        for (usize c = 0; c < width; ++c) {
            out_col_major[c * height + r] = line_out[c];
        }
    }
    return ImageStatus::kOk;
}

ImageStatus idct_2d(ArrayRef<f64> in_col_major, usize height, usize width,
                    MutArrayRef<f64> out_col_major, Allocator* alloc) {
    const usize total = height * width;
    if (in_col_major.size() != total || out_col_major.size() != total) {
        return ImageStatus::kInvalidArgument;
    }
    if (total == 0) {
        return ImageStatus::kOk;
    }
    Allocator* a = alloc ? alloc : default_allocator();
    const usize max_dim = max(height, width);
    Array<f64> tmp(a);
    tmp.resize(total);
    Array<f64> line_in(a);
    line_in.resize(max_dim);
    Array<f64> line_out(a);
    line_out.resize(max_dim);
    Array<c64> zbuf(a);
    zbuf.resize(max_dim);
    Array<c64> ibuf(a);
    ibuf.resize(max_dim);
    Array<c64> tw(a);
    tw.resize(max_dim);

    // Pass 1: columns (length = height).
    fill_dct_twiddles(height, true, MutArrayRef<c64>(tw.data(), height));
    for (usize c = 0; c < width; ++c) {
        ArrayRef<f64> col_in(in_col_major.data() + c * height, height);
        MutArrayRef<f64> col_out(tmp.data() + c * height, height);
        idct1d_makhoul(col_in, col_out, MutArrayRef<c64>(zbuf.data(), height),
                       MutArrayRef<c64>(ibuf.data(), height), ArrayRef<c64>(tw.data(), height));
    }

    // Pass 2: rows (length = width).
    fill_dct_twiddles(width, true, MutArrayRef<c64>(tw.data(), width));
    for (usize r = 0; r < height; ++r) {
        for (usize c = 0; c < width; ++c) {
            line_in[c] = tmp[c * height + r];
        }
        idct1d_makhoul(ArrayRef<f64>(line_in.data(), width),
                       MutArrayRef<f64>(line_out.data(), width),
                       MutArrayRef<c64>(zbuf.data(), width), MutArrayRef<c64>(ibuf.data(), width),
                       ArrayRef<c64>(tw.data(), width));
        for (usize c = 0; c < width; ++c) {
            out_col_major[c * height + r] = line_out[c];
        }
    }
    return ImageStatus::kOk;
}

ImageStatus integral_image_2d(ArrayRef<f64> in_col_major, usize height, usize width,
                              MutArrayRef<f64> out_sat_col_major) noexcept {
    const usize h1 = height + 1;
    const usize w1 = width + 1;
    if (in_col_major.size() != height * width || out_sat_col_major.size() != h1 * w1) {
        return ImageStatus::kInvalidArgument;
    }
    fill(out_sat_col_major.begin(), out_sat_col_major.end(), 0.0);
    if (height == 0 || width == 0) {
        return ImageStatus::kOk;
    }
    for (usize c = 0; c < width; ++c) {
        const usize cb = (c + 1) * h1;
        const usize cl = c * h1;
        const usize in_col = c * height;
        for (usize r = 0; r < height; ++r) {
            const usize r1 = r + 1;
            out_sat_col_major[cb + r1] = out_sat_col_major[cl + r1] + out_sat_col_major[cb + r]
                                       - out_sat_col_major[cl + r] + in_col_major[in_col + r];
        }
    }
    return ImageStatus::kOk;
}

ImageStatus integral_image_3d(ArrayRef<f64> in_col_major, usize height, usize width, usize depth,
                              MutArrayRef<f64> out_sat_col_major) noexcept {
    const usize h1 = height + 1;
    const usize w1 = width + 1;
    const usize d1 = depth + 1;
    const usize plane1 = h1 * w1;
    const usize plane_in = height * width;
    if (in_col_major.size() != plane_in * depth || out_sat_col_major.size() != plane1 * d1) {
        return ImageStatus::kInvalidArgument;
    }
    fill(out_sat_col_major.begin(), out_sat_col_major.end(), 0.0);
    if (height == 0 || width == 0 || depth == 0) {
        return ImageStatus::kOk;
    }
    auto idx = [&](usize r, usize c, usize p) noexcept -> usize {
        return p * plane1 + c * h1 + r;
    };
    for (usize p = 0; p < depth; ++p) {
        const usize p1 = p + 1;
        for (usize c = 0; c < width; ++c) {
            const usize c1 = c + 1;
            for (usize r = 0; r < height; ++r) {
                const usize r1 = r + 1;
                const f64 v = in_col_major[p * plane_in + c * height + r];
                out_sat_col_major[idx(r1, c1, p1)] =
                    out_sat_col_major[idx(r, c1, p1)] + out_sat_col_major[idx(r1, c, p1)]
                    + out_sat_col_major[idx(r1, c1, p)] - out_sat_col_major[idx(r, c, p1)]
                    - out_sat_col_major[idx(r, c1, p)] - out_sat_col_major[idx(r1, c, p)]
                    + out_sat_col_major[idx(r, c, p)] + v;
            }
        }
    }
    return ImageStatus::kOk;
}

ImageStatus phantom_ellipses(PhantomModel model, MutArrayRef<f64> out_10x6_col_major) noexcept {
    if (out_10x6_col_major.size() != 60) {
        return ImageStatus::kInvalidArgument;
    }
    const f64 (*src)[6] = (model == PhantomModel::kSheppLogan) ? kSheppLogan : kModSheppLogan;
    for (usize r = 0; r < 10; ++r) {
        for (usize c = 0; c < 6; ++c) {
            out_10x6_col_major[c * 10 + r] = src[r][c];
        }
    }
    return ImageStatus::kOk;
}

ImageStatus phantom_from_ellipses(ArrayRef<f64> ellipses_col_major, usize num_ellipses, usize n,
                                  MutArrayRef<f64> out_col_major, Allocator* alloc) {
    if (ellipses_col_major.size() != num_ellipses * 6 || out_col_major.size() != n * n) {
        return ImageStatus::kInvalidArgument;
    }
    fill(out_col_major.begin(), out_col_major.end(), 0.0);
    if (n == 0 || num_ellipses == 0) {
        return ImageStatus::kOk;
    }
    Allocator* a = alloc ? alloc : default_allocator();
    Array<f64> xvals(a);
    xvals.resize(n);
    if (n == 1) {
        xvals[0] = -1.0;
    } else {
        const f64 inv = 2.0 / as<f64>(n - 1);
        for (usize i = 0; i < n; ++i) {
            xvals[i] = -1.0 + as<f64>(i) * inv;
        }
    }
    auto eat = [&](usize r, usize c) noexcept -> f64 {
        return ellipses_col_major[c * num_ellipses + r];
    };
    for (usize k = 0; k < num_ellipses; ++k) {
        const f64 intensity = eat(k, 0);
        const f64 a_axis = eat(k, 1);
        const f64 b_axis = eat(k, 2);
        const f64 x0 = eat(k, 3);
        const f64 y0 = eat(k, 4);
        const f64 phi = eat(k, 5) * kPi / 180.0;
        const f64 a2 = a_axis * a_axis;
        const f64 b2 = b_axis * b_axis;
        if (!(a2 > 0.0) || !(b2 > 0.0) || !is_finite(a2) || !is_finite(b2)) {
            continue;
        }
        const f64 cos_p = cos(phi);
        const f64 sin_p = sin(phi);
        for (usize c = 0; c < n; ++c) {
            const f64 x = xvals[c] - x0;
            const usize col_off = c * n;
            for (usize r = 0; r < n; ++r) {
                const f64 y = xvals[n - 1 - r] - y0;
                const f64 u = x * cos_p + y * sin_p;
                const f64 v = y * cos_p - x * sin_p;
                if ((u * u) / a2 + (v * v) / b2 <= 1.0) {
                    out_col_major[col_off + r] += intensity;
                }
            }
        }
    }
    return ImageStatus::kOk;
}

ImageStatus checkerboard(usize side, usize tiles_y, usize tiles_x, MutArrayRef<f64> out_col_major,
                         Allocator* alloc) {
    const usize h = 2 * tiles_y * side;
    const usize w = 2 * tiles_x * side;
    if (out_col_major.size() != h * w) {
        return ImageStatus::kInvalidArgument;
    }
    if (h == 0 || w == 0 || side == 0) {
        return ImageStatus::kOk;
    }
    Allocator* a = alloc ? alloc : default_allocator();
    const usize s2 = 2 * side;
    Array<f64> x(a);
    x.resize(s2);
    if (s2 == 1) {
        x[0] = -1.0;
    } else {
        const f64 inv = 2.0 / as<f64>(s2 - 1);
        for (usize i = 0; i < s2; ++i) {
            x[i] = -1.0 + as<f64>(i) * inv;
        }
    }
    const usize half_w = w / 2;
    for (usize c = 0; c < w; ++c) {
        const f64 xc = x[c % s2];
        const f64 dim = (c >= half_w) ? 0.7 : 1.0;
        const usize col_off = c * h;
        for (usize r = 0; r < h; ++r) {
            const f64 xr = x[r % s2];
            out_col_major[col_off + r] = ((xr * xc < 0.0) ? 1.0 : 0.0) * dim;
        }
    }
    return ImageStatus::kOk;
}

ImageStatus psf_to_otf(ArrayRef<f64> psf_col_major, usize in_h, usize in_w, usize out_h,
                       usize out_w, MutArrayRef<c64> out_otf_col_major, Allocator* alloc) {
    if (in_h == 0 || in_w == 0 || out_h < in_h || out_w < in_w) {
        return ImageStatus::kInvalidArgument;
    }
    if (psf_col_major.size() != in_h * in_w || out_otf_col_major.size() != out_h * out_w) {
        return ImageStatus::kInvalidArgument;
    }
    Allocator* a = alloc ? alloc : default_allocator();
    const usize out_n = out_h * out_w;
    Array<c64> shifted(a);
    shifted.assign(out_n, c64{0.0, 0.0});
    const i64 shift_r = -as<i64>(in_h / 2);
    const i64 shift_c = -as<i64>(in_w / 2);
    for (usize c = 0; c < in_w; ++c) {
        const usize dst_c = wrap_shift(c, shift_c, out_w);
        for (usize r = 0; r < in_h; ++r) {
            const usize dst_r = wrap_shift(r, shift_r, out_h);
            shifted[dst_c * out_h + dst_r] = c64{psf_col_major[c * in_h + r], 0.0};
        }
    }
    Array<c64> tmp(a);
    tmp.resize(out_n);
    fft2_col_major(shifted.as_ref(), out_otf_col_major, out_h, out_w,
                   keel::fft::FftDirection::Forward, tmp);
    return ImageStatus::kOk;
}

ImageStatus otf_to_psf(ArrayRef<c64> otf_col_major, usize in_h, usize in_w, usize out_h,
                       usize out_w, MutArrayRef<c64> out_psf_col_major, Allocator* alloc) {
    if (in_h == 0 || in_w == 0 || out_h == 0 || out_w == 0 || out_h > in_h || out_w > in_w) {
        return ImageStatus::kInvalidArgument;
    }
    if (otf_col_major.size() != in_h * in_w || out_psf_col_major.size() != out_h * out_w) {
        return ImageStatus::kInvalidArgument;
    }
    Allocator* a = alloc ? alloc : default_allocator();
    const usize in_n = in_h * in_w;
    Array<c64> inv(a);
    inv.resize(in_n);
    Array<c64> tmp(a);
    tmp.resize(in_n);
    fft2_col_major(otf_col_major, inv.as_mut_ref(), in_h, in_w,
                   keel::fft::FftDirection::Inverse, tmp);

    // Circular shift by +floor(outsize / 2) and crop top-left out_h x out_w.
    const i64 shift_r = as<i64>(out_h / 2);
    const i64 shift_c = as<i64>(out_w / 2);
    for (usize c = 0; c < out_w; ++c) {
        const usize src_c = wrap_shift(c, -shift_c, in_w);
        for (usize r = 0; r < out_h; ++r) {
            const usize src_r = wrap_shift(r, -shift_r, in_h);
            out_psf_col_major[c * out_h + r] = inv[src_c * in_h + src_r];
        }
    }
    return ImageStatus::kOk;
}

ImageStatus normxcorr2(ArrayRef<f64> templ_col_major, usize th, usize tw,
                       ArrayRef<f64> img_col_major, usize ih, usize iw,
                       MutArrayRef<f64> out_col_major, Allocator* alloc) {
    if (th == 0 || tw == 0 || ih == 0 || iw == 0) {
        return ImageStatus::kInvalidArgument;
    }
    const usize out_h = ih + th - 1;
    const usize out_w = iw + tw - 1;
    if (templ_col_major.size() != th * tw || img_col_major.size() != ih * iw
        || out_col_major.size() != out_h * out_w) {
        return ImageStatus::kInvalidArgument;
    }
    Allocator* a = alloc ? alloc : default_allocator();
    const usize tn = th * tw;
    const usize in_n = ih * iw;

    // 1. Center template `a = templ - mean(templ)` and compute sumsq(a).
    f64 sum_t = 0.0;
    for (usize i = 0; i < tn; ++i) {
        sum_t += templ_col_major[i];
    }
    const f64 mean_t = sum_t / as<f64>(tn);
    Array<f64> a_cent(a);
    a_cent.resize(tn);
    f64 sumsq_a = 0.0;
    for (usize i = 0; i < tn; ++i) {
        const f64 v = templ_col_major[i] - mean_t;
        a_cent[i] = v;
        sumsq_a += v * v;
    }
    if (!(sumsq_a > 0.0)) {
        fill(out_col_major.begin(), out_col_major.end(), 0.0);
        return ImageStatus::kOk;
    }

    // 2. Numerator: full 2D convolution of `img` with `rot180(a_cent)` via 2D FFT.
    const usize fft_h = next_pow2(out_h);
    const usize fft_w = next_pow2(out_w);
    const usize fft_n = fft_h * fft_w;
    Array<c64> fa(a);
    fa.assign(fft_n, c64{0.0, 0.0});
    Array<c64> fb(a);
    fb.assign(fft_n, c64{0.0, 0.0});
    Array<c64> tmp(a);
    tmp.resize(fft_n);

    for (usize c = 0; c < tw; ++c) {
        for (usize r = 0; r < th; ++r) {
            fa[(tw - 1 - c) * fft_h + (th - 1 - r)] = c64{a_cent[c * th + r], 0.0};
        }
    }
    for (usize c = 0; c < iw; ++c) {
        for (usize r = 0; r < ih; ++r) {
            fb[c * fft_h + r] = c64{img_col_major[c * ih + r], 0.0};
        }
    }

    Array<c64> fa_f(a);
    fa_f.resize(fft_n);
    Array<c64> fb_f(a);
    fb_f.resize(fft_n);
    fft2_col_major(fa.as_ref(), fa_f.as_mut_ref(), fft_h, fft_w,
                   keel::fft::FftDirection::Forward, tmp);
    fft2_col_major(fb.as_ref(), fb_f.as_mut_ref(), fft_h, fft_w,
                   keel::fft::FftDirection::Forward, tmp);
    for (usize i = 0; i < fft_n; ++i) {
        fa_f[i] = fa_f[i] * fb_f[i];
    }
    fft2_col_major(fa_f.as_ref(), fb_f.as_mut_ref(), fft_h, fft_w,
                   keel::fft::FftDirection::Inverse, tmp);

    // 3. Denominator: exact local sums `sum_b` and `sum_b_sq` via 2D integral images.
    const usize h1 = ih + 1;
    const usize w1 = iw + 1;
    Array<f64> sat_b(a);
    sat_b.resize(h1 * w1);
    Array<f64> b_sq(a);
    b_sq.resize(in_n);
    for (usize i = 0; i < in_n; ++i) {
        b_sq[i] = img_col_major[i] * img_col_major[i];
    }
    Array<f64> sat_b2(a);
    sat_b2.resize(h1 * w1);
    (void)integral_image_2d(img_col_major, ih, iw, sat_b.as_mut_ref());
    (void)integral_image_2d(b_sq.as_ref(), ih, iw, sat_b2.as_mut_ref());

    auto rect_sum = [&](const Array<f64>& sat, usize r0, usize r1, usize c0,
                        usize c1) noexcept -> f64 {
        return sat[c1 * h1 + r1] - sat[c0 * h1 + r1] - sat[c1 * h1 + r0] + sat[c0 * h1 + r0];
    };

    const f64 inv_tn = 1.0 / as<f64>(tn);
    Array<f64> denoms(a);
    denoms.assign(out_h * out_w, 0.0);
    f64 max_denom = 0.0;
    for (usize c = 0; c < out_w; ++c) {
        const usize c0 = (c + 1 >= tw) ? (c + 1 - tw) : 0;
        const usize c1 = min(c + 1, iw);
        for (usize r = 0; r < out_h; ++r) {
            const usize r0 = (r + 1 >= th) ? (r + 1 - th) : 0;
            const usize r1 = min(r + 1, ih);
            const f64 sb = rect_sum(sat_b, r0, r1, c0, c1);
            const f64 sb2 = rect_sum(sat_b2, r0, r1, c0, c1);
            f64 var_b = sb2 - sb * sb * inv_tn;
            if (var_b < 0.0) {
                var_b = 0.0;
            }
            const f64 denom = sqrt(var_b * sumsq_a);
            denoms[c * out_h + r] = denom;
            if (denom > max_denom) {
                max_denom = denom;
            }
        }
    }
    const f64 denom_tol = 1000.0 * Limits<f64>::kEpsilon * max(1.0, max_denom);
    for (usize c = 0; c < out_w; ++c) {
        for (usize r = 0; r < out_h; ++r) {
            const usize out_idx = c * out_h + r;
            const f64 denom = denoms[out_idx];
            const f64 num = fb_f[c * fft_h + r].re;
            f64 v = (denom > denom_tol) ? (num / denom) : 0.0;
            if (!is_finite(v)) {
                v = 0.0;
            }
            out_col_major[out_idx] = clamp(v, -1.0, 1.0);
        }
    }
    return ImageStatus::kOk;
}

namespace {

template <typename Fn>
f64 fminbnd_brent(Fn f, f64 ax, f64 bx, f64 tol, i32 maxiter) noexcept {
    const f64 seps = sqrt(Limits<f64>::kEpsilon);
    const f64 c = 0.5 * (3.0 - sqrt(5.0));
    f64 a = ax;
    f64 b = bx;
    f64 v = a + c * (b - a);
    f64 w = v;
    f64 xf = v;
    f64 d = 0.0;
    f64 e = 0.0;
    f64 fx = f(xf);
    f64 fv = fx;
    f64 fw = fx;
    i32 iter = 0;
    f64 xm = 0.5 * (a + b);
    f64 tol1 = seps * abs(xf) + tol / 3.0;
    f64 tol2 = 2.0 * tol1;

    while (abs(xf - xm) > tol2 - 0.5 * (b - a)) {
        if (++iter > maxiter) {
            break;
        }
        bool gold = true;
        if (abs(e) > tol1) {
            const f64 r = (xf - w) * (fx - fv);
            f64 q = (xf - v) * (fx - fw);
            f64 p = (xf - v) * q - (xf - w) * r;
            q = 2.0 * (q - r);
            if (q > 0.0) {
                p = -p;
            }
            q = abs(q);
            const f64 etemp = e;
            e = d;
            if (abs(p) < abs(0.5 * q * etemp) && p > q * (a - xf) && p < q * (b - xf)) {
                d = p / q;
                const f64 u = xf + d;
                if (u - a < tol2 || b - u < tol2) {
                    const f64 si = (xm - xf >= 0.0) ? 1.0 : -1.0;
                    d = tol1 * si;
                }
                gold = false;
            }
        }
        if (gold) {
            e = (xf >= xm) ? (a - xf) : (b - xf);
            d = c * e;
        }
        const f64 si = (d >= 0.0) ? 1.0 : -1.0;
        const f64 u = xf + ((abs(d) >= tol1) ? d : tol1 * si);
        const f64 fu = f(u);
        if (fu <= fx) {
            if (u >= xf) {
                a = xf;
            } else {
                b = xf;
            }
            v = w;
            fv = fw;
            w = xf;
            fw = fx;
            xf = u;
            fx = fu;
        } else {
            if (u < xf) {
                a = u;
            } else {
                b = u;
            }
            if (fu <= fw || w == xf) {
                v = w;
                fv = fw;
                w = u;
                fw = fu;
            } else if (fu <= fv || v == xf || v == w) {
                v = u;
                fv = fu;
            }
        }
        xm = 0.5 * (a + b);
        tol1 = seps * abs(xf) + tol / 3.0;
        tol2 = 2.0 * tol1;
    }
    return xf;
}

} // namespace

ImageStatus deconvwnr_2d(ArrayRef<f64> img_col_major, usize height, usize width,
                         ArrayRef<f64> psf_col_major, usize psf_h, usize psf_w,
                         ArrayRef<f64> ncorr_col_major, ArrayRef<f64> icorr_col_major,
                         MutArrayRef<f64> out_col_major, Allocator* alloc) {
    const usize plane = height * width;
    if (height == 0 || width == 0 || psf_h == 0 || psf_w == 0 || psf_h > height || psf_w > width) {
        return ImageStatus::kInvalidArgument;
    }
    if (img_col_major.size() != plane || psf_col_major.size() != psf_h * psf_w
        || out_col_major.size() != plane) {
        return ImageStatus::kInvalidArgument;
    }
    const bool su_scalar = (ncorr_col_major.size() == 1);
    const bool sx_scalar = (icorr_col_major.size() == 1);
    if ((!su_scalar && ncorr_col_major.size() != plane)
        || (!sx_scalar && icorr_col_major.size() != plane)) {
        return ImageStatus::kInvalidArgument;
    }
    if (su_scalar && (!is_finite(ncorr_col_major[0]) || ncorr_col_major[0] < 0.0)) {
        return ImageStatus::kInvalidArgument;
    }
    if (sx_scalar && (!is_finite(icorr_col_major[0]) || icorr_col_major[0] < 0.0)) {
        return ImageStatus::kInvalidArgument;
    }

    Allocator* a = alloc ? alloc : default_allocator();
    Array<c64> h_otf(a);
    h_otf.resize(plane);
    const ImageStatus st =
        psf_to_otf(psf_col_major, psf_h, psf_w, height, width, h_otf.as_mut_ref(), a);
    if (st != ImageStatus::kOk) {
        return st;
    }

    Array<c64> tmp(a);
    tmp.resize(plane);
    Array<c64> cbuf(a);
    cbuf.resize(plane);
    Array<c64> fbuf(a);
    fbuf.resize(plane);

    Array<f64> su(a);
    if (su_scalar) {
        su.push_back(ncorr_col_major[0]);
    } else {
        su.resize(plane);
        for (usize i = 0; i < plane; ++i) {
            cbuf[i] = c64{ncorr_col_major[i], 0.0};
        }
        fft2_col_major(cbuf.as_ref(), fbuf.as_mut_ref(), height, width,
                       keel::fft::FftDirection::Forward, tmp);
        for (usize i = 0; i < plane; ++i) {
            su[i] = sqrt(fbuf[i].re * fbuf[i].re + fbuf[i].im * fbuf[i].im);
        }
    }

    Array<f64> sx(a);
    if (sx_scalar) {
        sx.push_back(icorr_col_major[0]);
    } else {
        sx.resize(plane);
        for (usize i = 0; i < plane; ++i) {
            cbuf[i] = c64{icorr_col_major[i], 0.0};
        }
        fft2_col_major(cbuf.as_ref(), fbuf.as_mut_ref(), height, width,
                       keel::fft::FftDirection::Forward, tmp);
        for (usize i = 0; i < plane; ++i) {
            sx[i] = sqrt(fbuf[i].re * fbuf[i].re + fbuf[i].im * fbuf[i].im);
        }
    }

    for (usize i = 0; i < plane; ++i) {
        cbuf[i] = c64{img_col_major[i], 0.0};
    }
    fft2_col_major(cbuf.as_ref(), fbuf.as_mut_ref(), height, width,
                   keel::fft::FftDirection::Forward, tmp);

    const f64 sqrt_eps = sqrt(Limits<f64>::kEpsilon);
    for (usize i = 0; i < plane; ++i) {
        const c64 hk = h_otf[i];
        const f64 m2 = hk.re * hk.re + hk.im * hk.im;
        const f64 sx_i = sx_scalar ? sx[0] : sx[i];
        const f64 su_i = su_scalar ? su[0] : su[i];
        f64 denom = m2 * sx_i + su_i;
        if (denom < sqrt_eps) {
            denom = sqrt_eps;
        }
        const f64 g_scale = sx_i / denom;
        const c64 gk{hk.re * g_scale, -hk.im * g_scale};
        cbuf[i] = gk * fbuf[i];
    }

    fft2_col_major(cbuf.as_ref(), fbuf.as_mut_ref(), height, width,
                   keel::fft::FftDirection::Inverse, tmp);
    for (usize i = 0; i < plane; ++i) {
        out_col_major[i] = fbuf[i].re;
    }
    return ImageStatus::kOk;
}

ImageStatus edgetaper_2d(ArrayRef<f64> img_col_major, usize height, usize width,
                         ArrayRef<f64> psf_col_major, usize psf_h, usize psf_w,
                         MutArrayRef<f64> out_col_major, Allocator* alloc) {
    const usize plane = height * width;
    if (plane < 2 || psf_h * psf_w < 2 || 2 * psf_h > height || 2 * psf_w > width) {
        return ImageStatus::kInvalidArgument;
    }
    if (img_col_major.size() != plane || psf_col_major.size() != psf_h * psf_w
        || out_col_major.size() != plane) {
        return ImageStatus::kInvalidArgument;
    }

    Allocator* a = alloc ? alloc : default_allocator();
    f64 img_min = Limits<f64>::infinity();
    f64 img_max = -Limits<f64>::infinity();
    for (usize i = 0; i < plane; ++i) {
        if (img_col_major[i] < img_min) {
            img_min = img_col_major[i];
        }
        if (img_col_major[i] > img_max) {
            img_max = img_col_major[i];
        }
    }

    auto compute_beta = [&](bool along_rows, usize size_n) -> Array<f64> {
        const usize proj_len = along_rows ? psf_h : psf_w;
        Array<f64> proj(a);
        proj.assign(proj_len, 0.0);
        if (along_rows) {
            for (usize c = 0; c < psf_w; ++c) {
                for (usize r = 0; r < psf_h; ++r) {
                    proj[r] += psf_col_major[c * psf_h + r];
                }
            }
        } else {
            for (usize c = 0; c < psf_w; ++c) {
                for (usize r = 0; r < psf_h; ++r) {
                    proj[c] += psf_col_major[c * psf_h + r];
                }
            }
        }
        const usize l = size_n - 1;
        Array<f64> padded(a);
        padded.assign(l, 0.0);
        for (usize k = 0; k < l && k < proj_len; ++k) {
            padded[k] = proj[k];
        }
        Array<c64> f_out(a);
        f_out.resize(l);
        keel::fft::fft_1d_real(padded.as_ref(), f_out.as_mut_ref());
        Array<c64> z_in(a);
        z_in.resize(l);
        for (usize k = 0; k < l; ++k) {
            z_in[k] = c64{f_out[k].re * f_out[k].re + f_out[k].im * f_out[k].im, 0.0};
        }
        Array<c64> z_out(a);
        z_out.resize(l);
        keel::fft::ifft_1d(z_in.as_ref(), z_out.as_mut_ref());
        Array<f64> beta(a);
        beta.resize(size_n);
        f64 zmax = 0.0;
        for (usize k = 0; k < l; ++k) {
            beta[k] = z_out[k].re;
            if (beta[k] > zmax) {
                zmax = beta[k];
            }
        }
        beta[l] = beta[0];
        if (zmax > 0.0) {
            for (usize k = 0; k < size_n; ++k) {
                beta[k] /= zmax;
            }
        }
        return beta;
    };

    Array<f64> beta_r = compute_beta(true, height);
    Array<f64> beta_c = compute_beta(false, width);

    Array<c64> h_otf(a);
    h_otf.resize(plane);
    const ImageStatus st =
        psf_to_otf(psf_col_major, psf_h, psf_w, height, width, h_otf.as_mut_ref(), a);
    if (st != ImageStatus::kOk) {
        return st;
    }

    Array<c64> cbuf(a);
    cbuf.resize(plane);
    Array<c64> fbuf(a);
    fbuf.resize(plane);
    Array<c64> tmp(a);
    tmp.resize(plane);
    for (usize i = 0; i < plane; ++i) {
        cbuf[i] = c64{img_col_major[i], 0.0};
    }
    fft2_col_major(cbuf.as_ref(), fbuf.as_mut_ref(), height, width,
                   keel::fft::FftDirection::Forward, tmp);
    for (usize i = 0; i < plane; ++i) {
        cbuf[i] = fbuf[i] * h_otf[i];
    }
    fft2_col_major(cbuf.as_ref(), fbuf.as_mut_ref(), height, width,
                   keel::fft::FftDirection::Inverse, tmp);

    for (usize c = 0; c < width; ++c) {
        const f64 one_minus_bc = 1.0 - beta_c[c];
        const usize col_off = c * height;
        for (usize r = 0; r < height; ++r) {
            const usize idx = col_off + r;
            const f64 alpha = (1.0 - beta_r[r]) * one_minus_bc;
            f64 v = alpha * img_col_major[idx] + (1.0 - alpha) * fbuf[idx].re;
            if (v < img_min) {
                v = img_min;
            }
            if (v > img_max) {
                v = img_max;
            }
            out_col_major[idx] = v;
        }
    }
    return ImageStatus::kOk;
}

ImageStatus deconvreg_2d(ArrayRef<f64> img_col_major, usize height, usize width,
                         ArrayRef<f64> psf_col_major, usize psf_h, usize psf_w, f64 noise_power,
                         f64 lambda_lo, f64 lambda_hi, ArrayRef<f64> regop_col_major, usize regop_h,
                         usize regop_w, MutArrayRef<f64> out_col_major, f64& lambda_out,
                         Allocator* alloc) {
    const usize plane = height * width;
    if (height < 3 || width < 3 || psf_h * psf_w < 2 || psf_h > height || psf_w > width) {
        return ImageStatus::kInvalidArgument;
    }
    if (!is_finite(noise_power) || noise_power < 0.0 || !is_finite(lambda_lo)
        || !is_finite(lambda_hi) || lambda_lo < 0.0 || lambda_hi < lambda_lo) {
        return ImageStatus::kInvalidArgument;
    }
    if (img_col_major.size() != plane || psf_col_major.size() != psf_h * psf_w
        || out_col_major.size() != plane) {
        return ImageStatus::kInvalidArgument;
    }
    const bool use_default_regop = (regop_h == 0 && regop_w == 0 && regop_col_major.size() == 0);
    if (!use_default_regop
        && (regop_h == 0 || regop_w == 0 || regop_h > height || regop_w > width
            || regop_col_major.size() != regop_h * regop_w)) {
        return ImageStatus::kInvalidArgument;
    }

    bool all_zero = true;
    for (usize i = 0; i < psf_col_major.size(); ++i) {
        if (psf_col_major[i] != 0.0) {
            all_zero = false;
            break;
        }
    }
    if (all_zero) {
        return ImageStatus::kInvalidArgument;
    }

    Allocator* a = alloc ? alloc : default_allocator();
    Array<c64> h_otf(a);
    h_otf.resize(plane);
    ImageStatus st = psf_to_otf(psf_col_major, psf_h, psf_w, height, width, h_otf.as_mut_ref(), a);
    if (st != ImageStatus::kOk) {
        return st;
    }

    Array<c64> r_otf(a);
    r_otf.resize(plane);
    if (use_default_regop) {
        constexpr f64 kLaplacian[9] = {0.0, 1.0, 0.0, 1.0, -4.0, 1.0, 0.0, 1.0, 0.0};
        st = psf_to_otf(ArrayRef<f64>(kLaplacian, 9), 3, 3, height, width, r_otf.as_mut_ref(), a);
    } else {
        st = psf_to_otf(regop_col_major, regop_h, regop_w, height, width, r_otf.as_mut_ref(), a);
    }
    if (st != ImageStatus::kOk) {
        return st;
    }

    Array<f64> h2(a);
    h2.resize(plane);
    Array<f64> r2(a);
    r2.resize(plane);
    for (usize i = 0; i < plane; ++i) {
        h2[i] = h_otf[i].re * h_otf[i].re + h_otf[i].im * h_otf[i].im;
        r2[i] = r_otf[i].re * r_otf[i].re + r_otf[i].im * r_otf[i].im;
    }

    Array<c64> cbuf(a);
    cbuf.resize(plane);
    Array<c64> fi(a);
    fi.resize(plane);
    Array<c64> tmp(a);
    tmp.resize(plane);
    for (usize i = 0; i < plane; ++i) {
        cbuf[i] = c64{img_col_major[i], 0.0};
    }
    fft2_col_major(cbuf.as_ref(), fi.as_mut_ref(), height, width,
                   keel::fft::FftDirection::Forward, tmp);

    const f64 seps = sqrt(Limits<f64>::kEpsilon);
    f64 lagra = lambda_lo;
    if (lambda_lo < lambda_hi) {
        Array<f64> r4g2(a);
        r4g2.resize(plane);
        Array<f64> h4(a);
        h4.resize(plane);
        Array<f64> r4(a);
        r4.resize(plane);
        Array<f64> h2r22(a);
        h2r22.resize(plane);
        for (usize i = 0; i < plane; ++i) {
            const f64 abs_fi = sqrt(fi[i].re * fi[i].re + fi[i].im * fi[i].im);
            const f64 t = r2[i] * abs_fi;
            r4g2[i] = t * t;
            h4[i] = h2[i] * h2[i];
            r4[i] = r2[i] * r2[i];
            h2r22[i] = 2.0 * h2[i] * r2[i];
        }
        const f64 scaled_np = noise_power * as<f64>(plane);
        auto res_offset = [&](f64 lam) noexcept -> f64 {
            f64 s = 0.0;
            for (usize i = 0; i < plane; ++i) {
                const f64 denom = h4[i] + lam * h2r22[i] + lam * lam * r4[i] + seps;
                s += r4g2[i] / denom;
            }
            return abs(lam * lam * s - scaled_np);
        };
        lagra = fminbnd_brent(res_offset, lambda_lo, lambda_hi, 1e-4, 500);
    }
    lambda_out = lagra;

    for (usize i = 0; i < plane; ++i) {
        f64 denom = h2[i] + lagra * r2[i];
        if (denom < seps) {
            denom = seps;
        }
        const f64 inv = 1.0 / denom;
        const c64 g{h_otf[i].re * inv, -h_otf[i].im * inv};
        cbuf[i] = g * fi[i];
    }

    Array<c64> jc(a);
    jc.resize(plane);
    fft2_col_major(cbuf.as_ref(), jc.as_mut_ref(), height, width,
                   keel::fft::FftDirection::Inverse, tmp);
    for (usize i = 0; i < plane; ++i) {
        out_col_major[i] = jc[i].re;
    }
    return ImageStatus::kOk;
}

} // namespace keel::image
