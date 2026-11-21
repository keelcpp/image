#include "keel/image.h"

#include "keel/fft.h"
#include "keel/math.h"

namespace keel::image {

namespace {

template <typename T>
void rot90_2(ArrayRef<T> src, usize h, usize w, MutArrayRef<T> dst) noexcept {
    for (usize c = 0; c < w; ++c) {
        for (usize r = 0; r < h; ++r) {
            dst[c * h + r] = src[(w - 1 - c) * h + (h - 1 - r)];
        }
    }
}

template <typename T>
void fftshift_2d(ArrayRef<T> src, usize h, usize w, MutArrayRef<T> dst) noexcept {
    const usize hh = (h + 1) / 2;
    const usize ww = (w + 1) / 2;
    for (usize c = 0; c < w; ++c) {
        const usize cs = (c + ww) % w;
        for (usize r = 0; r < h; ++r) {
            const usize rs = (r + hh) % h;
            dst[c * h + r] = src[cs * h + rs];
        }
    }
}

template <typename T>
void ifftshift_2d(ArrayRef<T> src, usize h, usize w, MutArrayRef<T> dst) noexcept {
    const usize hh = h / 2;
    const usize ww = w / 2;
    for (usize c = 0; c < w; ++c) {
        const usize cs = (c + ww) % w;
        for (usize r = 0; r < h; ++r) {
            const usize rs = (r + hh) % h;
            dst[c * h + r] = src[cs * h + rs];
        }
    }
}

template <typename T>
void bilinear_interp_2d(ArrayRef<T> hd, usize hh, usize wh, usize m_out, usize n_out,
                        MutArrayRef<T> out) noexcept {
    const f64 dr_src = 2.0 / as<f64>(hh);
    const f64 dc_src = 2.0 / as<f64>(wh);
    const f64 dr_dst = 2.0 / as<f64>(m_out);
    const f64 dc_dst = 2.0 / as<f64>(n_out);
    for (usize c = 0; c < n_out; ++c) {
        const f64 cx = (as<f64>(c) - as<f64>(n_out - 1) / 2.0) * dc_dst;
        const f64 sf_c = (cx + 1.0) / dc_src - 0.5;
        const i64 c0 = max<i64>(0, as<i64>(floor(sf_c)));
        const i64 c1 = min<i64>(as<i64>(wh) - 1, c0 + 1);
        const f64 ac = sf_c - as<f64>(c0);
        for (usize r = 0; r < m_out; ++r) {
            const f64 cy = (as<f64>(r) - as<f64>(m_out - 1) / 2.0) * dr_dst;
            const f64 sf_r = (cy + 1.0) / dr_src - 0.5;
            const i64 r0 = max<i64>(0, as<i64>(floor(sf_r)));
            const i64 r1 = min<i64>(as<i64>(hh) - 1, r0 + 1);
            const f64 ar = sf_r - as<f64>(r0);
            const T v00 = hd[as<usize>(c0) * hh + as<usize>(r0)];
            const T v01 = hd[as<usize>(c0) * hh + as<usize>(r1)];
            const T v10 = hd[as<usize>(c1) * hh + as<usize>(r0)];
            const T v11 = hd[as<usize>(c1) * hh + as<usize>(r1)];
            const T v0 = v00 + ar * (v01 - v00);
            const T v1 = v10 + ar * (v11 - v10);
            out[c * m_out + r] = v0 + ac * (v1 - v0);
        }
    }
}

template <typename T>
T linear_interp_uniform(ArrayRef<f64> xs, ArrayRef<T> ys, f64 q) noexcept {
    if (q <= xs[0]) return ys[0];
    if (q >= xs[xs.size() - 1]) return ys[ys.size() - 1];
    const f64 dx = xs[1] - xs[0];
    const f64 t = (q - xs[0]) / dx;
    const usize i = as<usize>(floor(t));
    const f64 a = t - as<f64>(i);
    return (1.0 - a) * ys[i] + a * ys[i + 1];
}

void conv2_full(ArrayRef<f64> a, usize ha, usize wa, ArrayRef<f64> b, usize hb, usize wb,
                MutArrayRef<f64> out) noexcept {
    const usize ho = ha + hb - 1;
    const usize wo = wa + wb - 1;
    for (usize i = 0; i < ho * wo; ++i) out[i] = 0.0;
    for (usize ca = 0; ca < wa; ++ca) {
        for (usize ra = 0; ra < ha; ++ra) {
            const f64 va = a[ca * ha + ra];
            if (va == 0.0) continue;
            for (usize cb = 0; cb < wb; ++cb) {
                const usize co = ca + cb;
                for (usize rb = 0; rb < hb; ++rb) {
                    out[co * ho + (ra + rb)] += va * b[cb * hb + rb];
                }
            }
        }
    }
}

} // namespace

ImageStatus fsamp2_real(ArrayRef<f64> hd_col_major, usize h, usize w,
                        MutArrayRef<f64> out_col_major, Allocator* alloc) {
    const usize n = h * w;
    if (h == 0 || w == 0 || hd_col_major.size() != n || out_col_major.size() != n) {
        return ImageStatus::kInvalidArgument;
    }

    // 1. ifftshift (equivalent to rot90(fftshift(rot90(Hd, 2)), 2)).
    Array<c64> shifted(n, c64(0.0, 0.0), alloc);
    {
        const usize hh = h / 2;
        const usize ww = w / 2;
        for (usize c = 0; c < w; ++c) {
            const usize cs = (c + ww) % w;
            for (usize r = 0; r < h; ++r) {
                const usize rs = (r + hh) % h;
                shifted[c * h + r] = c64(hd_col_major[cs * h + rs], 0.0);
            }
        }
    }

    // 2. 2-D inverse FFT (axis 0 then axis 1).
    Array<c64> tmp(n, c64(0.0, 0.0), alloc);
    const usize dims[2] = {h, w};
    const fft::FftBatchLayout lay0 = fft::FftBatchLayout::from_col_major(ArrayRef<usize>(dims, 2), 0);
    const fft::FftBatchLayout lay1 = fft::FftBatchLayout::from_col_major(ArrayRef<usize>(dims, 2), 1);
    fft::ifft_batch(shifted.as_ref(), tmp.as_mut_ref(), lay0);
    fft::ifft_batch(tmp.as_ref(), shifted.as_mut_ref(), lay1);

    // 3. Real part -> fftshift -> rot90(h, 2).
    Array<f64> real_part(n, 0.0, alloc);
    for (usize i = 0; i < n; ++i) real_part[i] = shifted[i].real();
    Array<f64> h_shifted(n, 0.0, alloc);
    fftshift_2d(real_part.as_ref(), h, w, h_shifted.as_mut_ref());
    rot90_2(h_shifted.as_ref(), h, w, out_col_major);
    return ImageStatus::kOk;
}

ImageStatus fsamp2_complex(ArrayRef<c64> hd_col_major, usize h, usize w,
                           MutArrayRef<c64> out_col_major, Allocator* alloc) {
    const usize n = h * w;
    if (h == 0 || w == 0 || hd_col_major.size() != n || out_col_major.size() != n) {
        return ImageStatus::kInvalidArgument;
    }

    Array<f64> re_in(n, 0.0, alloc);
    Array<f64> im_in(n, 0.0, alloc);
    Array<f64> re_out(n, 0.0, alloc);
    Array<f64> im_out(n, 0.0, alloc);
    for (usize i = 0; i < n; ++i) {
        re_in[i] = hd_col_major[i].real();
        im_in[i] = hd_col_major[i].imag();
    }
    (void)fsamp2_real(re_in.as_ref(), h, w, re_out.as_mut_ref(), alloc);
    (void)fsamp2_real(im_in.as_ref(), h, w, im_out.as_mut_ref(), alloc);
    for (usize i = 0; i < n; ++i) {
        out_col_major[i] = c64(re_out[i], im_out[i]);
    }
    return ImageStatus::kOk;
}

ImageStatus ftrans2(ArrayRef<f64> b, ArrayRef<f64> t_col_major, usize mt, usize nt,
                    Array<f64>& out_col_major, usize& out_h, usize& out_w, Allocator* alloc) {
    const usize len = b.size();
    if (len == 0 || (len % 2) == 0) return ImageStatus::kInvalidArgument;

    bool all_zero = true;
    for (usize i = 0; i < len; ++i) {
        if (b[i] != 0.0) {
            all_zero = false;
            break;
        }
    }
    if (all_zero) return ImageStatus::kInvalidArgument;

    const f64 sqrt_eps = sqrt(Limits<f64>::kEpsilon);
    for (usize i = 0; i < len; ++i) {
        if (abs(b[i] - b[len - 1 - i]) > sqrt_eps) return ImageStatus::kInvalidArgument;
    }

    Array<f64> t_buf(alloc);
    if (t_col_major.empty()) {
        mt = 3;
        nt = 3;
        t_buf.resize(9, 0.0);
        t_buf[0] = 1.0 / 8.0;
        t_buf[1] = 2.0 / 8.0;
        t_buf[2] = 1.0 / 8.0;
        t_buf[3] = 2.0 / 8.0;
        t_buf[4] = -4.0 / 8.0;
        t_buf[5] = 2.0 / 8.0;
        t_buf[6] = 1.0 / 8.0;
        t_buf[7] = 2.0 / 8.0;
        t_buf[8] = 1.0 / 8.0;
    } else {
        if (mt == 0 || nt == 0 || t_col_major.size() != mt * nt) {
            return ImageStatus::kInvalidArgument;
        }
        bool t_zero = true;
        t_buf.resize(mt * nt, 0.0);
        for (usize i = 0; i < mt * nt; ++i) {
            t_buf[i] = t_col_major[i];
            if (t_buf[i] != 0.0) t_zero = false;
        }
        if (t_zero) return ImageStatus::kInvalidArgument;
    }

    const usize n = (len - 1) / 2;
    const usize shift = len / 2;
    Array<f64> a(n + 1, 0.0, alloc);
    a[0] = b[shift % len];
    for (usize k = 1; k <= n; ++k) {
        a[k] = 2.0 * b[(k + shift) % len];
    }

    if (n == 0) {
        out_h = 1;
        out_w = 1;
        out_col_major.resize(1);
        out_col_major[0] = a[0];
        return ImageStatus::kOk;
    }

    const usize inset_r = (mt - 1) / 2;
    const usize inset_c = (nt - 1) / 2;

    Array<f64> p0(1, 1.0, alloc);
    usize hp0 = 1;
    usize wp0 = 1;
    Array<f64> p1(mt * nt, 0.0, alloc);
    for (usize i = 0; i < mt * nt; ++i) p1[i] = t_buf[i];
    usize hp1 = mt;
    usize wp1 = nt;

    Array<f64> h_mat(mt * nt, 0.0, alloc);
    usize hh = mt;
    usize wh = nt;
    for (usize i = 0; i < mt * nt; ++i) h_mat[i] = p1[i] * a[1];
    h_mat[inset_c * hh + inset_r] += a[0];

    usize cur_rows_off = inset_r;
    usize cur_cols_off = inset_c;
    for (usize i = 2; i <= n; ++i) {
        const usize hp2 = mt + hp1 - 1;
        const usize wp2 = nt + wp1 - 1;
        Array<f64> p2(hp2 * wp2, 0.0, alloc);
        conv2_full(t_buf.as_ref(), mt, nt, p1.as_ref(), hp1, wp1, p2.as_mut_ref());
        for (usize k = 0; k < hp2 * wp2; ++k) p2[k] *= 2.0;

        cur_rows_off += inset_r;
        cur_cols_off += inset_c;
        for (usize c = 0; c < wp0; ++c) {
            for (usize r = 0; r < hp0; ++r) {
                p2[(cur_cols_off + c) * hp2 + (cur_rows_off + r)] -= p0[c * hp0 + r];
            }
        }

        Array<f64> h_new(hp2 * wp2, 0.0, alloc);
        for (usize k = 0; k < hp2 * wp2; ++k) h_new[k] = p2[k] * a[i];
        for (usize c = 0; c < wh; ++c) {
            for (usize r = 0; r < hh; ++r) {
                h_new[(inset_c + c) * hp2 + (inset_r + r)] += h_mat[c * hh + r];
            }
        }

        h_mat = keel::move(h_new);
        hh = hp2;
        wh = wp2;
        p0 = keel::move(p1);
        hp0 = hp1;
        wp0 = wp1;
        p1 = keel::move(p2);
        hp1 = hp2;
        wp1 = wp2;
    }

    out_h = hh;
    out_w = wh;
    out_col_major.resize(hh * wh);
    rot90_2(h_mat.as_ref(), hh, wh, out_col_major.as_mut_ref());
    return ImageStatus::kOk;
}

ImageStatus fwind1_real(ArrayRef<f64> hd_col_major, usize hh, usize wh, ArrayRef<f64> win1,
                        ArrayRef<f64> win2, MutArrayRef<f64> out_col_major, Allocator* alloc) {
    if (hh == 0 || wh == 0 || hd_col_major.size() != hh * wh) return ImageStatus::kInvalidArgument;
    const usize n = win1.size();
    if (n < 2) return ImageStatus::kInvalidArgument;

    const usize m_out = win2.empty() ? n : win2.size();
    const usize n_out = n;
    if (out_col_major.size() != m_out * n_out) return ImageStatus::kInvalidArgument;

    Array<f64> w_mat(m_out * n_out, 0.0, alloc);
    if (win2.empty()) {
        const f64 sqrt_eps = sqrt(Limits<f64>::kEpsilon);
        for (usize i = 0; i < n; ++i) {
            if (abs(win1[i] - win1[n - 1 - i]) > sqrt_eps) return ImageStatus::kInvalidArgument;
        }
        Array<f64> t(n, 0.0, alloc);
        for (usize i = 0; i < n; ++i) {
            t[i] = (as<f64>(i) - as<f64>(n - 1) / 2.0) * (2.0 / as<f64>(n - 1));
        }
        for (usize c = 0; c < n; ++c) {
            for (usize r = 0; r < n; ++r) {
                const f64 rr = sqrt(t[c] * t[c] + t[r] * t[r]);
                w_mat[c * n + r] =
                    (rr < t[0] || rr > t[n - 1]) ? 0.0 : linear_interp_uniform<f64>(t.as_ref(), win1, rr);
            }
        }
    } else {
        for (usize c = 0; c < n; ++c) {
            for (usize r = 0; r < m_out; ++r) {
                w_mat[c * m_out + r] = win2[r] * win1[c];
            }
        }
    }

    Array<f64> hd_interp(alloc);
    ArrayRef<f64> hd_used = hd_col_major;
    if (hh != m_out || wh != n_out) {
        hd_interp.resize(m_out * n_out, 0.0);
        bilinear_interp_2d<f64>(hd_col_major, hh, wh, m_out, n_out, hd_interp.as_mut_ref());
        hd_used = hd_interp.as_ref();
    }

    Array<f64> h(m_out * n_out, 0.0, alloc);
    (void)fsamp2_real(hd_used, m_out, n_out, h.as_mut_ref(), alloc);
    for (usize i = 0; i < m_out * n_out; ++i) {
        out_col_major[i] = h[i] * w_mat[i];
    }
    return ImageStatus::kOk;
}

ImageStatus fwind1_complex(ArrayRef<c64> hd_col_major, usize hh, usize wh, ArrayRef<c64> win1,
                           ArrayRef<c64> win2, MutArrayRef<c64> out_col_major, Allocator* alloc) {
    if (hh == 0 || wh == 0 || hd_col_major.size() != hh * wh) return ImageStatus::kInvalidArgument;
    const usize n = win1.size();
    if (n < 2) return ImageStatus::kInvalidArgument;

    const usize m_out = win2.empty() ? n : win2.size();
    const usize n_out = n;
    if (out_col_major.size() != m_out * n_out) return ImageStatus::kInvalidArgument;

    Array<c64> w_mat(m_out * n_out, c64(0.0, 0.0), alloc);
    if (win2.empty()) {
        const f64 sqrt_eps = sqrt(Limits<f64>::kEpsilon);
        for (usize i = 0; i < n; ++i) {
            if (abs(win1[i] - win1[n - 1 - i]) > sqrt_eps) return ImageStatus::kInvalidArgument;
        }
        Array<f64> t(n, 0.0, alloc);
        for (usize i = 0; i < n; ++i) {
            t[i] = (as<f64>(i) - as<f64>(n - 1) / 2.0) * (2.0 / as<f64>(n - 1));
        }
        for (usize c = 0; c < n; ++c) {
            for (usize r = 0; r < n; ++r) {
                const f64 rr = sqrt(t[c] * t[c] + t[r] * t[r]);
                w_mat[c * n + r] = (rr < t[0] || rr > t[n - 1])
                                       ? c64(0.0, 0.0)
                                       : linear_interp_uniform<c64>(t.as_ref(), win1, rr);
            }
        }
    } else {
        for (usize c = 0; c < n; ++c) {
            for (usize r = 0; r < m_out; ++r) {
                w_mat[c * m_out + r] = win2[r] * win1[c];
            }
        }
    }

    Array<c64> hd_interp(alloc);
    ArrayRef<c64> hd_used = hd_col_major;
    if (hh != m_out || wh != n_out) {
        hd_interp.resize(m_out * n_out, c64(0.0, 0.0));
        bilinear_interp_2d<c64>(hd_col_major, hh, wh, m_out, n_out, hd_interp.as_mut_ref());
        hd_used = hd_interp.as_ref();
    }

    Array<c64> h(m_out * n_out, c64(0.0, 0.0), alloc);
    (void)fsamp2_complex(hd_used, m_out, n_out, h.as_mut_ref(), alloc);
    for (usize i = 0; i < m_out * n_out; ++i) {
        out_col_major[i] = h[i] * w_mat[i];
    }
    return ImageStatus::kOk;
}

ImageStatus fwind2_real(ArrayRef<f64> hd_col_major, usize hh, usize wh,
                        ArrayRef<f64> win_col_major, usize hw, usize ww,
                        MutArrayRef<f64> out_col_major, Allocator* alloc) {
    if (hh == 0 || wh == 0 || hd_col_major.size() != hh * wh) return ImageStatus::kInvalidArgument;
    if (hw == 0 || ww == 0 || win_col_major.size() != hw * ww) return ImageStatus::kInvalidArgument;

    Array<f64> w_outer(alloc);
    ArrayRef<f64> w_ref = win_col_major;
    usize mw = hw;
    usize nw = ww;
    if (hw == 1 || ww == 1) {
        const usize n = hw * ww;
        mw = n;
        nw = n;
        w_outer.resize(n * n, 0.0);
        for (usize c = 0; c < n; ++c) {
            for (usize r = 0; r < n; ++r) {
                w_outer[c * n + r] = win_col_major[r] * win_col_major[c];
            }
        }
        w_ref = w_outer.as_ref();
    }
    if (out_col_major.size() != mw * nw) return ImageStatus::kInvalidArgument;

    Array<f64> hd_interp(alloc);
    ArrayRef<f64> hd_used = hd_col_major;
    if (hh != mw || wh != nw) {
        hd_interp.resize(mw * nw, 0.0);
        bilinear_interp_2d<f64>(hd_col_major, hh, wh, mw, nw, hd_interp.as_mut_ref());
        hd_used = hd_interp.as_ref();
    }

    Array<f64> h(mw * nw, 0.0, alloc);
    (void)fsamp2_real(hd_used, mw, nw, h.as_mut_ref(), alloc);
    for (usize i = 0; i < mw * nw; ++i) {
        out_col_major[i] = h[i] * w_ref[i];
    }
    return ImageStatus::kOk;
}

ImageStatus fwind2_complex(ArrayRef<c64> hd_col_major, usize hh, usize wh,
                           ArrayRef<c64> win_col_major, usize hw, usize ww,
                           MutArrayRef<c64> out_col_major, Allocator* alloc) {
    if (hh == 0 || wh == 0 || hd_col_major.size() != hh * wh) return ImageStatus::kInvalidArgument;
    if (hw == 0 || ww == 0 || win_col_major.size() != hw * ww) return ImageStatus::kInvalidArgument;

    Array<c64> w_outer(alloc);
    ArrayRef<c64> w_ref = win_col_major;
    usize mw = hw;
    usize nw = ww;
    if (hw == 1 || ww == 1) {
        const usize n = hw * ww;
        mw = n;
        nw = n;
        w_outer.resize(n * n, c64(0.0, 0.0));
        for (usize c = 0; c < n; ++c) {
            for (usize r = 0; r < n; ++r) {
                w_outer[c * n + r] = win_col_major[r] * win_col_major[c];
            }
        }
        w_ref = w_outer.as_ref();
    }
    if (out_col_major.size() != mw * nw) return ImageStatus::kInvalidArgument;

    Array<c64> hd_interp(alloc);
    ArrayRef<c64> hd_used = hd_col_major;
    if (hh != mw || wh != nw) {
        hd_interp.resize(mw * nw, c64(0.0, 0.0));
        bilinear_interp_2d<c64>(hd_col_major, hh, wh, mw, nw, hd_interp.as_mut_ref());
        hd_used = hd_interp.as_ref();
    }

    Array<c64> h(mw * nw, c64(0.0, 0.0), alloc);
    (void)fsamp2_complex(hd_used, mw, nw, h.as_mut_ref(), alloc);
    for (usize i = 0; i < mw * nw; ++i) {
        out_col_major[i] = h[i] * w_ref[i];
    }
    return ImageStatus::kOk;
}

} // namespace keel::image
