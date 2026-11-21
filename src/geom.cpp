#include "keel/image.h"

#include "keel/math.h"

namespace keel::image {

namespace {

using KernelFn = f64 (*)(f64) noexcept;

inline f64 k_box(f64 x) noexcept {
    return (x >= -0.5 && x < 0.5) ? 1.0 : 0.0;
}

inline f64 k_triangle(f64 x) noexcept {
    const f64 ax = abs(x);
    return (ax < 1.0) ? (1.0 - ax) : 0.0;
}

inline f64 k_cubic(f64 x) noexcept {
    const f64 a = abs(x);
    if (a < 1.0) return (1.5 * a - 2.5) * a * a + 1.0;
    if (a < 2.0) return ((-0.5 * a + 2.5) * a - 4.0) * a + 2.0;
    return 0.0;
}

inline f64 sinc_pi(f64 x) noexcept {
    if (abs(x) < 1e-12) return 1.0;
    const f64 px = kPi * x;
    return sin(px) / px;
}

inline f64 k_lanczos2(f64 x) noexcept {
    return (abs(x) < 2.0) ? (sinc_pi(x) * sinc_pi(x / 2.0)) : 0.0;
}

inline f64 k_lanczos3(f64 x) noexcept {
    return (abs(x) < 3.0) ? (sinc_pi(x) * sinc_pi(x / 3.0)) : 0.0;
}

inline void resolve_kernel(ResizeKernel k, KernelFn& fn, f64& width) noexcept {
    switch (k) {
        case ResizeKernel::kBox:
            fn = k_box;
            width = 1.0;
            return;
        case ResizeKernel::kTriangle:
            fn = k_triangle;
            width = 2.0;
            return;
        case ResizeKernel::kCubic:
            fn = k_cubic;
            width = 4.0;
            return;
        case ResizeKernel::kLanczos2:
            fn = k_lanczos2;
            width = 4.0;
            return;
        case ResizeKernel::kLanczos3:
            fn = k_lanczos3;
            width = 6.0;
            return;
        default:
            fn = nullptr;
            width = 0.0;
            return;
    }
}

inline i32 mirror_idx(i32 i, i32 n) noexcept {
    if (n <= 1) return 0;
    const i32 cycle = 2 * n;
    i32 m = ((i % cycle) + cycle) % cycle;
    if (m >= n) m = cycle - 1 - m;
    return m;
}

inline i32 sym_idx(i32 i, i32 n) noexcept {
    if (n <= 1) return 0;
    if (i < 0) return -1 - i;
    if (i >= n) return 2 * n - 1 - i;
    return i;
}

struct ContribTable {
    usize taps = 0;
    Array<i32> first_idx;
    Array<f64> weights; // row-major out_len x taps

    explicit ContribTable(Allocator* alloc) : first_idx(alloc), weights(alloc) {}
};

ContribTable build_contrib_table(usize out_len, f64 scale, KernelFn fn, f64 kw, bool aa,
                                 Allocator* alloc) {
    f64 scl = 1.0;
    if (aa && scale < 1.0) {
        scl = scale;
        kw = kw / scale;
    }
    const usize p = as<usize>(ceil(kw)) + 2;
    ContribTable tbl(alloc);
    tbl.taps = p;
    tbl.first_idx.resize(out_len, i32{0});
    tbl.weights.resize(out_len * p, 0.0);

    for (usize o = 0; o < out_len; ++o) {
        const f64 u = (as<f64>(o) + 1.0) / scale + 0.5 * (1.0 - 1.0 / scale);
        const i32 left = as<i32>(floor(u - kw / 2.0));
        tbl.first_idx[o] = left - 1;
        f64 sum = 0.0;
        f64* wrow = tbl.weights.data() + o * p;
        for (usize j = 0; j < p; ++j) {
            const f64 w = scl * fn(scl * (u - as<f64>(left + as<i32>(j))));
            wrow[j] = w;
            sum += w;
        }
        if (sum != 0.0) {
            for (usize j = 0; j < p; ++j) wrow[j] /= sum;
        }
    }
    return tbl;
}

inline f64 sample_zero_oob(ArrayRef<f64> img, usize h, usize w, usize plane_off, i32 y,
                           i32 x) noexcept {
    if (y < 0 || x < 0 || y >= as<i32>(h) || x >= as<i32>(w)) return 0.0;
    return img[plane_off + as<usize>(x) * h + as<usize>(y)];
}

inline f64 sample_bilinear(ArrayRef<f64> img, usize h, usize w, usize plane_off, f64 y,
                           f64 x) noexcept {
    const i32 y0 = as<i32>(floor(y));
    const i32 x0 = as<i32>(floor(x));
    const f64 dy = y - as<f64>(y0);
    const f64 dx = x - as<f64>(x0);
    const f64 v00 = sample_zero_oob(img, h, w, plane_off, y0, x0);
    const f64 v01 = sample_zero_oob(img, h, w, plane_off, y0, x0 + 1);
    const f64 v10 = sample_zero_oob(img, h, w, plane_off, y0 + 1, x0);
    const f64 v11 = sample_zero_oob(img, h, w, plane_off, y0 + 1, x0 + 1);
    return v00 * (1.0 - dy) * (1.0 - dx) + v01 * (1.0 - dy) * dx +
           v10 * dy * (1.0 - dx) + v11 * dy * dx;
}

// IEEE-754 round-half-to-even (matches std::nearbyint under default rounding).
inline i32 round_half_to_even_i32(f64 x) noexcept {
    const f64 f = floor(x);
    const f64 frac = x - f;
    if (frac < 0.5) return as<i32>(f);
    if (frac > 0.5) return as<i32>(f + 1.0);
    const i32 fi = as<i32>(f);
    return ((fi & 1) == 0) ? fi : (fi + 1);
}

inline f64 snap_near_int(f64 v) noexcept {
    const f64 r = round(v);
    return (abs(v - r) < 1e-9) ? r : v;
}

} // namespace

ImageStatus resize_2d(ArrayRef<f64> in_col_major, usize in_h, usize in_w, usize channels,
                      usize out_h, usize out_w, f64 scale_y, f64 scale_x, ResizeKernel kernel,
                      bool antialias, MutArrayRef<f64> out_col_major, Allocator* alloc) {
    if (in_col_major.size() != in_h * in_w * channels) return ImageStatus::kInvalidArgument;
    if (out_col_major.size() != out_h * out_w * channels) return ImageStatus::kInvalidArgument;
    if (out_h == 0 || out_w == 0 || channels == 0) return ImageStatus::kOk;
    if (in_h == 0 || in_w == 0) return ImageStatus::kInvalidArgument;
    if (!(scale_y > 0.0) || !(scale_x > 0.0)) return ImageStatus::kInvalidArgument;

    if (kernel == ResizeKernel::kNearest) {
        const f64 sx = as<f64>(in_w) / as<f64>(out_w);
        const f64 sy = as<f64>(in_h) / as<f64>(out_h);
        const i32 ih = as<i32>(in_h);
        const i32 iw = as<i32>(in_w);
        for (usize c = 0; c < channels; ++c) {
            const usize in_off = c * in_h * in_w;
            const usize out_off = c * out_h * out_w;
            for (usize yo = 0; yo < out_h; ++yo) {
                const f64 ys = (as<f64>(yo) + 0.5) * sy - 0.5;
                i32 yi = as<i32>(floor(ys + 0.5));
                if (yi < 0) yi = 0;
                if (yi >= ih) yi = ih - 1;
                for (usize xo = 0; xo < out_w; ++xo) {
                    const f64 xs = (as<f64>(xo) + 0.5) * sx - 0.5;
                    i32 xi = as<i32>(floor(xs + 0.5));
                    if (xi < 0) xi = 0;
                    if (xi >= iw) xi = iw - 1;
                    out_col_major[out_off + xo * out_h + yo] =
                        in_col_major[in_off + as<usize>(xi) * in_h + as<usize>(yi)];
                }
            }
        }
        return ImageStatus::kOk;
    }

    KernelFn fn = nullptr;
    f64 kw = 0.0;
    resolve_kernel(kernel, fn, kw);
    const ContribTable cy = build_contrib_table(out_h, scale_y, fn, kw, antialias, alloc);
    const ContribTable cx = build_contrib_table(out_w, scale_x, fn, kw, antialias, alloc);

    const i32 ih = as<i32>(in_h);
    const i32 iw = as<i32>(in_w);
    Array<f64> tmp(out_h * in_w, 0.0, alloc);
    for (usize c = 0; c < channels; ++c) {
        const usize in_off = c * in_h * in_w;
        const usize out_off = c * out_h * out_w;
        for (usize x = 0; x < in_w; ++x) {
            for (usize y = 0; y < out_h; ++y) {
                const i32 first = cy.first_idx[y];
                const f64* wrow = cy.weights.data() + y * cy.taps;
                f64 v = 0.0;
                for (usize j = 0; j < cy.taps; ++j) {
                    const i32 yi = mirror_idx(first + as<i32>(j), ih);
                    v += wrow[j] * in_col_major[in_off + x * in_h + as<usize>(yi)];
                }
                tmp[y + x * out_h] = v;
            }
        }
        for (usize x = 0; x < out_w; ++x) {
            const i32 first = cx.first_idx[x];
            const f64* wrow = cx.weights.data() + x * cx.taps;
            for (usize y = 0; y < out_h; ++y) {
                f64 v = 0.0;
                for (usize j = 0; j < cx.taps; ++j) {
                    const i32 xi = mirror_idx(first + as<i32>(j), iw);
                    v += wrow[j] * tmp[y + as<usize>(xi) * out_h];
                }
                out_col_major[out_off + x * out_h + y] = v;
            }
        }
    }
    return ImageStatus::kOk;
}

ImageStatus resize_3d(ArrayRef<f64> in_col_major, usize in_r, usize in_c, usize in_d,
                      usize out_r, usize out_c, usize out_d, f64 scale_y, f64 scale_x,
                      f64 scale_z, ResizeKernel kernel, bool antialias,
                      MutArrayRef<f64> out_col_major, Allocator* alloc) {
    if (in_col_major.size() != in_r * in_c * in_d) return ImageStatus::kInvalidArgument;
    if (out_col_major.size() != out_r * out_c * out_d) return ImageStatus::kInvalidArgument;
    if (out_r == 0 || out_c == 0 || out_d == 0) return ImageStatus::kOk;
    if (in_r == 0 || in_c == 0 || in_d == 0) return ImageStatus::kInvalidArgument;
    if (!(scale_y > 0.0) || !(scale_x > 0.0) || !(scale_z > 0.0)) {
        return ImageStatus::kInvalidArgument;
    }

    const i32 ir = as<i32>(in_r);
    const i32 ic = as<i32>(in_c);
    const i32 id = as<i32>(in_d);

    if (kernel == ResizeKernel::kNearest) {
        auto u_coord = [](f64 o1, f64 sc) noexcept {
            return o1 / sc + 0.5 * (1.0 - 1.0 / sc);
        };
        for (usize z = 0; z < out_d; ++z) {
            const f64 uz = u_coord(as<f64>(z) + 1.0, scale_z);
            const i32 zi = mirror_idx(as<i32>(floor(uz + 0.5)) - 1, id);
            for (usize x = 0; x < out_c; ++x) {
                const f64 ux = u_coord(as<f64>(x) + 1.0, scale_x);
                const i32 xi = mirror_idx(as<i32>(floor(ux + 0.5)) - 1, ic);
                for (usize y = 0; y < out_r; ++y) {
                    const f64 uy = u_coord(as<f64>(y) + 1.0, scale_y);
                    const i32 yi = mirror_idx(as<i32>(floor(uy + 0.5)) - 1, ir);
                    out_col_major[y + x * out_r + z * out_r * out_c] =
                        in_col_major[as<usize>(yi) + as<usize>(xi) * in_r +
                                     as<usize>(zi) * in_r * in_c];
                }
            }
        }
        return ImageStatus::kOk;
    }

    KernelFn fn = nullptr;
    f64 kw = 0.0;
    resolve_kernel(kernel, fn, kw);
    const ContribTable cy = build_contrib_table(out_r, scale_y, fn, kw, antialias, alloc);
    const ContribTable cx = build_contrib_table(out_c, scale_x, fn, kw, antialias, alloc);
    const ContribTable cz = build_contrib_table(out_d, scale_z, fn, kw, antialias, alloc);

    Array<f64> tmp1(out_r * in_c * in_d, 0.0, alloc);
    for (usize z = 0; z < in_d; ++z) {
        for (usize x = 0; x < in_c; ++x) {
            for (usize y = 0; y < out_r; ++y) {
                const i32 first = cy.first_idx[y];
                const f64* wrow = cy.weights.data() + y * cy.taps;
                f64 s = 0.0;
                for (usize j = 0; j < cy.taps; ++j) {
                    const i32 yi = mirror_idx(first + as<i32>(j), ir);
                    s += wrow[j] * in_col_major[as<usize>(yi) + x * in_r + z * in_r * in_c];
                }
                tmp1[y + x * out_r + z * out_r * in_c] = s;
            }
        }
    }

    Array<f64> tmp2(out_r * out_c * in_d, 0.0, alloc);
    for (usize z = 0; z < in_d; ++z) {
        for (usize x = 0; x < out_c; ++x) {
            const i32 first = cx.first_idx[x];
            const f64* wrow = cx.weights.data() + x * cx.taps;
            for (usize y = 0; y < out_r; ++y) {
                f64 s = 0.0;
                for (usize j = 0; j < cx.taps; ++j) {
                    const i32 xi = mirror_idx(first + as<i32>(j), ic);
                    s += wrow[j] * tmp1[y + as<usize>(xi) * out_r + z * out_r * in_c];
                }
                tmp2[y + x * out_r + z * out_r * out_c] = s;
            }
        }
    }

    for (usize z = 0; z < out_d; ++z) {
        const i32 first = cz.first_idx[z];
        const f64* wrow = cz.weights.data() + z * cz.taps;
        for (usize x = 0; x < out_c; ++x) {
            for (usize y = 0; y < out_r; ++y) {
                f64 s = 0.0;
                for (usize j = 0; j < cz.taps; ++j) {
                    const i32 zi = mirror_idx(first + as<i32>(j), id);
                    s += wrow[j] * tmp2[y + x * out_r + as<usize>(zi) * out_r * out_c];
                }
                out_col_major[y + x * out_r + z * out_r * out_c] = s;
            }
        }
    }
    return ImageStatus::kOk;
}

void rotate_2d_dims(usize in_h, usize in_w, f64 angle_deg, bool crop, usize& out_h,
                    usize& out_w) noexcept {
    f64 a = fmod(angle_deg, 360.0);
    if (a < 0.0) a += 360.0;
    const i64 kk = as<i64>(round(a / 90.0));
    if (abs(a - as<f64>(kk) * 90.0) < 1e-9) {
        const i32 k = as<i32>(((kk % 4) + 4) % 4);
        const usize hl = (k % 2 == 0) ? in_h : in_w;
        const usize wl = (k % 2 == 0) ? in_w : in_h;
        out_h = crop ? in_h : hl;
        out_w = crop ? in_w : wl;
        return;
    }
    if (crop) {
        out_h = in_h;
        out_w = in_w;
        return;
    }
    const f64 rad = angle_deg * (kPi / 180.0);
    const f64 co = cos(rad);
    const f64 si = sin(rad);
    const f64 hd = as<f64>(in_h);
    const f64 wd = as<f64>(in_w);
    const f64 raw_h = abs(hd * co) + abs(wd * si);
    const f64 raw_w = abs(wd * co) + abs(hd * si);
    out_h = as<usize>(ceil(snap_near_int(raw_h)));
    out_w = as<usize>(ceil(snap_near_int(raw_w)));
    if (out_h < 1) out_h = 1;
    if (out_w < 1) out_w = 1;
}

ImageStatus rotate_2d(ArrayRef<f64> in_col_major, usize in_h, usize in_w, usize channels,
                      f64 angle_deg, bool nearest, bool crop, MutArrayRef<f64> out_col_major,
                      usize out_h, usize out_w) noexcept {
    if (in_col_major.size() != in_h * in_w * channels) return ImageStatus::kInvalidArgument;
    if (out_col_major.size() != out_h * out_w * channels) return ImageStatus::kInvalidArgument;
    if (out_h == 0 || out_w == 0 || channels == 0) return ImageStatus::kOk;

    f64 a = fmod(angle_deg, 360.0);
    if (a < 0.0) a += 360.0;
    const i64 kk = as<i64>(round(a / 90.0));
    if (abs(a - as<f64>(kk) * 90.0) < 1e-9) {
        const i32 k = as<i32>(((kk % 4) + 4) % 4);
        const usize hl = (k % 2 == 0) ? in_h : in_w;
        const usize wl = (k % 2 == 0) ? in_w : in_h;
        i64 row_off = 0;
        i64 col_off = 0;
        if (crop) {
            row_off = as<i64>(floor((as<f64>(hl) - as<f64>(in_h)) / 2.0));
            col_off = as<i64>(floor((as<f64>(wl) - as<f64>(in_w)) / 2.0));
            if (k == 3 && ((as<i64>(hl) - as<i64>(in_h)) & i64{1})) {
                row_off += 1;
            }
        }
        const i64 ih = as<i64>(in_h);
        const i64 iw = as<i64>(in_w);
        for (usize c = 0; c < channels; ++c) {
            const usize in_off = c * in_h * in_w;
            const usize out_off = c * out_h * out_w;
            for (usize i = 0; i < out_h; ++i) {
                for (usize j = 0; j < out_w; ++j) {
                    const i64 li = as<i64>(i) + row_off;
                    const i64 lj = as<i64>(j) + col_off;
                    f64 v = 0.0;
                    if (li >= 0 && li < as<i64>(hl) && lj >= 0 && lj < as<i64>(wl)) {
                        i64 sr = 0;
                        i64 sc = 0;
                        switch (k) {
                            case 0: sr = li; sc = lj; break;
                            case 1: sr = lj; sc = iw - 1 - li; break;
                            case 2: sr = ih - 1 - li; sc = iw - 1 - lj; break;
                            default: sr = ih - 1 - lj; sc = li; break;
                        }
                        v = sample_zero_oob(in_col_major, in_h, in_w, in_off, as<i32>(sr),
                                            as<i32>(sc));
                    }
                    out_col_major[out_off + j * out_h + i] = v;
                }
            }
        }
        return ImageStatus::kOk;
    }

    const f64 rad = angle_deg * (kPi / 180.0);
    const f64 co = cos(rad);
    const f64 si = sin(rad);
    const f64 cy_out = (as<f64>(out_h) - 1.0) / 2.0;
    const f64 cx_out = (as<f64>(out_w) - 1.0) / 2.0;
    const f64 cy_in = (as<f64>(in_h) - 1.0) / 2.0;
    const f64 cx_in = (as<f64>(in_w) - 1.0) / 2.0;

    for (usize c = 0; c < channels; ++c) {
        const usize in_off = c * in_h * in_w;
        const usize out_off = c * out_h * out_w;
        for (usize y = 0; y < out_h; ++y) {
            const f64 dy = as<f64>(y) - cy_out;
            for (usize x = 0; x < out_w; ++x) {
                const f64 dx = as<f64>(x) - cx_out;
                const f64 xs = co * dx - si * dy + cx_in;
                const f64 ys = si * dx + co * dy + cy_in;
                f64 v = 0.0;
                if (nearest) {
                    const i32 yi = round_half_to_even_i32(ys);
                    const i32 xi = round_half_to_even_i32(xs);
                    v = sample_zero_oob(in_col_major, in_h, in_w, in_off, yi, xi);
                } else {
                    v = sample_bilinear(in_col_major, in_h, in_w, in_off, ys, xs);
                }
                out_col_major[out_off + x * out_h + y] = v;
            }
        }
    }
    return ImageStatus::kOk;
}

ImageStatus translate_2d(ArrayRef<f64> in_col_major, usize h, usize w, usize channels, f64 dx,
                         f64 dy, MutArrayRef<f64> out_col_major) noexcept {
    if (in_col_major.size() != h * w * channels) return ImageStatus::kInvalidArgument;
    if (out_col_major.size() != h * w * channels) return ImageStatus::kInvalidArgument;
    if (h == 0 || w == 0 || channels == 0) return ImageStatus::kOk;

    for (usize c = 0; c < channels; ++c) {
        const usize off = c * h * w;
        for (usize y = 0; y < h; ++y) {
            const f64 ys = as<f64>(y) - dy;
            for (usize x = 0; x < w; ++x) {
                const f64 xs = as<f64>(x) - dx;
                out_col_major[off + x * h + y] = sample_bilinear(in_col_major, h, w, off, ys, xs);
            }
        }
    }
    return ImageStatus::kOk;
}

ImageStatus pyramid_reduce_2d(ArrayRef<f64> in_col_major, usize h, usize w, usize channels,
                              MutArrayRef<f64> out_col_major, Allocator* alloc) {
    static constexpr f64 kBurt[5] = {1.0 / 16.0, 4.0 / 16.0, 6.0 / 16.0, 4.0 / 16.0, 1.0 / 16.0};
    const usize out_h = (h + 1) / 2;
    const usize out_w = (w + 1) / 2;
    if (in_col_major.size() != h * w * channels) return ImageStatus::kInvalidArgument;
    if (out_col_major.size() != out_h * out_w * channels) return ImageStatus::kInvalidArgument;
    if (h == 0 || w == 0 || channels == 0) return ImageStatus::kOk;

    const i32 ih = as<i32>(h);
    const i32 iw = as<i32>(w);
    Array<f64> tmp(h * w, 0.0, alloc);
    for (usize c = 0; c < channels; ++c) {
        const usize in_off = c * h * w;
        const usize out_off = c * out_h * out_w;
        for (usize r = 0; r < h; ++r) {
            for (usize cc = 0; cc < w; ++cc) {
                f64 acc = 0.0;
                for (i32 k = -2; k <= 2; ++k) {
                    const i32 cs = sym_idx(as<i32>(cc) + k, iw);
                    acc += kBurt[k + 2] * in_col_major[in_off + as<usize>(cs) * h + r];
                }
                tmp[cc * h + r] = acc;
            }
        }
        for (usize yo = 0; yo < out_h; ++yo) {
            const usize ys = 2 * yo;
            for (usize xo = 0; xo < out_w; ++xo) {
                const usize xs = 2 * xo;
                f64 acc = 0.0;
                for (i32 k = -2; k <= 2; ++k) {
                    const i32 rs = sym_idx(as<i32>(ys) + k, ih);
                    acc += kBurt[k + 2] * tmp[xs * h + as<usize>(rs)];
                }
                out_col_major[out_off + xo * out_h + yo] = acc;
            }
        }
    }
    return ImageStatus::kOk;
}

ImageStatus pyramid_expand_2d(ArrayRef<f64> in_col_major, usize h, usize w, usize channels,
                              MutArrayRef<f64> out_col_major, Allocator* alloc) {
    static constexpr f64 kBurt[5] = {1.0 / 16.0, 4.0 / 16.0, 6.0 / 16.0, 4.0 / 16.0, 1.0 / 16.0};
    const usize out_h = h > 0 ? (2 * h - 1) : 0;
    const usize out_w = w > 0 ? (2 * w - 1) : 0;
    if (in_col_major.size() != h * w * channels) return ImageStatus::kInvalidArgument;
    if (out_col_major.size() != out_h * out_w * channels) return ImageStatus::kInvalidArgument;
    if (h == 0 || w == 0 || channels == 0) return ImageStatus::kOk;

    const i32 oh = as<i32>(out_h);
    const i32 ow = as<i32>(out_w);
    Array<f64> stuffed(out_h * out_w, 0.0, alloc);
    Array<f64> tmp(out_h * out_w, 0.0, alloc);
    for (usize c = 0; c < channels; ++c) {
        const usize in_off = c * h * w;
        const usize out_off = c * out_h * out_w;
        for (usize i = 0; i < out_h * out_w; ++i) stuffed[i] = 0.0;
        for (usize r = 0; r < h; ++r) {
            for (usize cc = 0; cc < w; ++cc) {
                stuffed[(2 * cc) * out_h + 2 * r] = in_col_major[in_off + cc * h + r];
            }
        }
        for (usize r = 0; r < out_h; ++r) {
            for (usize cc = 0; cc < out_w; ++cc) {
                f64 acc = 0.0;
                for (i32 k = -2; k <= 2; ++k) {
                    const i32 cs = as<i32>(cc) + k;
                    if (cs >= 0 && cs < ow) {
                        acc += kBurt[k + 2] * stuffed[as<usize>(cs) * out_h + r];
                    }
                }
                tmp[cc * out_h + r] = 2.0 * acc;
            }
        }
        for (usize cc = 0; cc < out_w; ++cc) {
            for (usize r = 0; r < out_h; ++r) {
                f64 acc = 0.0;
                for (i32 k = -2; k <= 2; ++k) {
                    const i32 rs = as<i32>(r) + k;
                    if (rs >= 0 && rs < oh) {
                        acc += kBurt[k + 2] * tmp[cc * out_h + as<usize>(rs)];
                    }
                }
                out_col_major[out_off + cc * out_h + r] = 2.0 * acc;
            }
        }
    }
    return ImageStatus::kOk;
}

} // namespace keel::image
