#include "keel/image.h"

#include "keel/algo.h"
#include "keel/math.h"

namespace keel::image {

namespace {

struct Sym3 {
    f64 m00, m01, m02, m11, m12, m22;
};

void jacobi_eigen_3x3(const Sym3& M, f64 evals[3], f64 evecs[9]) noexcept {
    f64 a[3][3] = {
        {M.m00, M.m01, M.m02},
        {M.m01, M.m11, M.m12},
        {M.m02, M.m12, M.m22},
    };
    f64 V[3][3] = {{1.0, 0.0, 0.0}, {0.0, 1.0, 0.0}, {0.0, 0.0, 1.0}};
    for (i32 sweep = 0; sweep < 50; ++sweep) {
        const f64 off = abs(a[0][1]) + abs(a[0][2]) + abs(a[1][2]);
        if (off < 1e-30) break;
        for (i32 p = 0; p < 2; ++p) {
            for (i32 q = p + 1; q < 3; ++q) {
                if (abs(a[p][q]) < 1e-30) continue;
                const f64 theta = (a[q][q] - a[p][p]) / (2.0 * a[p][q]);
                f64 t = 0.0;
                if (abs(theta) > 1e150) {
                    t = 1.0 / (2.0 * theta);
                } else {
                    t = (theta >= 0.0 ? 1.0 : -1.0) / (abs(theta) + sqrt(theta * theta + 1.0));
                }
                const f64 c = 1.0 / sqrt(t * t + 1.0);
                const f64 s = t * c;
                const f64 app = a[p][p];
                const f64 aqq = a[q][q];
                const f64 apq = a[p][q];
                a[p][p] = app - t * apq;
                a[q][q] = aqq + t * apq;
                a[p][q] = 0.0;
                a[q][p] = 0.0;
                for (i32 r = 0; r < 3; ++r) {
                    if (r != p && r != q) {
                        const f64 arp = a[r][p];
                        const f64 arq = a[r][q];
                        a[r][p] = c * arp - s * arq;
                        a[r][q] = s * arp + c * arq;
                        a[p][r] = a[r][p];
                        a[q][r] = a[r][q];
                    }
                }
                for (i32 r = 0; r < 3; ++r) {
                    const f64 vrp = V[r][p];
                    const f64 vrq = V[r][q];
                    V[r][p] = c * vrp - s * vrq;
                    V[r][q] = s * vrp + c * vrq;
                }
            }
        }
    }
    evals[0] = a[0][0];
    evals[1] = a[1][1];
    evals[2] = a[2][2];
    for (i32 k = 0; k < 3; ++k) {
        for (i32 i = 0; i < 3; ++i) {
            evecs[i + 3 * k] = V[i][k];
        }
    }
}

void collect_masked_channel(ArrayRef<f64> ch, ArrayRef<u8> mask, Array<f64>& out) {
    out.clear();
    const usize n = ch.size();
    out.reserve(n);
    if (mask.empty()) {
        for (usize i = 0; i < n; ++i) out.push_back(ch[i]);
    } else {
        for (usize i = 0; i < n; ++i) {
            if (mask[i] != 0) out.push_back(ch[i]);
        }
    }
}

inline f64 wrap_hue_rad(f64 a, f64 b) noexcept {
    if (a == 0.0 && b == 0.0) return 0.0;
    f64 h = atan2(b, a);
    if (h < 0.0) h += kTau;
    return h;
}

} // namespace

ImageStatus illumwhite_planar(ArrayRef<f64> r, ArrayRef<f64> g, ArrayRef<f64> b,
                              ArrayRef<u8> mask, f64 percentile, Color3& out,
                              Allocator* alloc) {
    const usize n = r.size();
    if (n == 0 || g.size() != n || b.size() != n) return ImageStatus::kInvalidArgument;
    if (!mask.empty() && mask.size() != n) return ImageStatus::kInvalidArgument;
    if (!is_finite(percentile) || percentile < 0.0 || percentile >= 100.0) {
        return ImageStatus::kInvalidArgument;
    }

    ArrayRef<f64> chans[3] = {r, g, b};
    f64 res[3] = {0.0, 0.0, 0.0};
    Array<f64> plane(alloc);
    for (usize c = 0; c < 3; ++c) {
        collect_masked_channel(chans[c], mask, plane);
        if (plane.empty()) return ImageStatus::kInvalidArgument;
        keel::sort(plane);
        const usize m = plane.size();
        usize k = as<usize>(floor(percentile * 0.01 * as<f64>(m)));
        if (k >= m) k = m - 1;
        res[c] = plane[m - k - 1];
    }
    out = Color3{res[0], res[1], res[2]};
    return ImageStatus::kOk;
}

ImageStatus illumgray_planar(ArrayRef<f64> r, ArrayRef<f64> g, ArrayRef<f64> b,
                             ArrayRef<u8> mask, f64 p_lo, f64 p_hi, f64 norm_exp,
                             Color3& out, Allocator* alloc) {
    const usize n = r.size();
    if (n == 0 || g.size() != n || b.size() != n) return ImageStatus::kInvalidArgument;
    if (!mask.empty() && mask.size() != n) return ImageStatus::kInvalidArgument;
    if (!is_finite(p_lo) || !is_finite(p_hi) || !is_finite(norm_exp)) {
        return ImageStatus::kInvalidArgument;
    }
    if (p_lo < 0.0 || p_lo >= 100.0 || p_hi < 0.0 || p_hi >= 100.0 || p_lo + p_hi > 100.0) {
        return ImageStatus::kInvalidArgument;
    }
    if (!(norm_exp > 0.0)) return ImageStatus::kInvalidArgument;

    ArrayRef<f64> chans[3] = {r, g, b};
    f64 res[3] = {0.0, 0.0, 0.0};
    Array<f64> plane(alloc);
    for (usize c = 0; c < 3; ++c) {
        collect_masked_channel(chans[c], mask, plane);
        if (plane.empty()) return ImageStatus::kInvalidArgument;
        keel::sort(plane);
        const usize m = plane.size();
        const usize k_lo = as<usize>(floor(p_lo * 0.01 * as<f64>(m)));
        const usize k_hi = as<usize>(floor(p_hi * 0.01 * as<f64>(m)));
        if (k_lo + k_hi >= m) return ImageStatus::kInvalidArgument;

        const usize lo = k_lo;
        const usize hi = m - k_hi;
        f64 sum = 0.0;
        f64 comp = 0.0; // Kahan compensation for high-precision accumulation
        for (usize k = lo; k < hi; ++k) {
            const f64 av = abs(plane[k]);
            const f64 term = (norm_exp == 1.0) ? av : pow(av, norm_exp);
            const f64 y = term - comp;
            const f64 t = sum + y;
            comp = (t - sum) - y;
            sum = t;
        }
        const f64 cnt = as<f64>(hi - lo);
        res[c] = (norm_exp == 1.0) ? (sum / cnt) : (pow(sum, 1.0 / norm_exp) / cnt);
    }
    out = Color3{res[0], res[1], res[2]};
    return ImageStatus::kOk;
}

ImageStatus illumpca_planar(ArrayRef<f64> r, ArrayRef<f64> g, ArrayRef<f64> b,
                            ArrayRef<u8> mask, f64 percentage, Color3& out,
                            Allocator* alloc) {
    const usize n = r.size();
    if (n == 0 || g.size() != n || b.size() != n) return ImageStatus::kInvalidArgument;
    if (!mask.empty() && mask.size() != n) return ImageStatus::kInvalidArgument;
    if (!is_finite(percentage) || !(percentage > 0.0) || percentage > 50.0) {
        return ImageStatus::kInvalidArgument;
    }

    Array<f64> rows(alloc);
    rows.reserve(3 * n);
    if (mask.empty()) {
        for (usize i = 0; i < n; ++i) {
            rows.push_back(r[i]);
            rows.push_back(g[i]);
            rows.push_back(b[i]);
        }
    } else {
        for (usize i = 0; i < n; ++i) {
            if (mask[i] == 0) continue;
            rows.push_back(r[i]);
            rows.push_back(g[i]);
            rows.push_back(b[i]);
        }
    }
    const usize m = rows.size() / 3;
    if (m == 0) return ImageStatus::kInvalidArgument;

    f64 a0[3] = {0.0, 0.0, 0.0};
    for (usize k = 0; k < m; ++k) {
        a0[0] += rows[3 * k + 0];
        a0[1] += rows[3 * k + 1];
        a0[2] += rows[3 * k + 2];
    }
    a0[0] /= as<f64>(m);
    a0[1] /= as<f64>(m);
    a0[2] /= as<f64>(m);
    const f64 norm_a0_sq = a0[0] * a0[0] + a0[1] * a0[1] + a0[2] * a0[2];
    if (!is_finite(norm_a0_sq)) return ImageStatus::kInvalidArgument;

    Array<usize> idx(m, usize{0}, alloc);
    Array<f64> proj(m, 0.0, alloc);
    if (norm_a0_sq > 0.0) {
        for (usize k = 0; k < m; ++k) {
            proj[k] = (rows[3 * k + 0] * a0[0] + rows[3 * k + 1] * a0[1] +
                       rows[3 * k + 2] * a0[2]) /
                      norm_a0_sq;
            idx[k] = k;
        }
    } else {
        for (usize k = 0; k < m; ++k) {
            proj[k] = 0.0;
            idx[k] = k;
        }
    }
    keel::sort(idx, [&](usize lhs, usize rhs) noexcept { return proj[lhs] < proj[rhs]; });

    Array<f64> selected(alloc);
    if (percentage >= 50.0 || m == 1) {
        selected.reserve(3 * m);
        for (usize k = 0; k < m; ++k) {
            const usize row = idx[k];
            selected.push_back(rows[3 * row + 0]);
            selected.push_back(rows[3 * row + 1]);
            selected.push_back(rows[3 * row + 2]);
        }
    } else {
        usize lo = as<usize>(floor(percentage * 0.01 * as<f64>(m)));
        if (lo < 1) lo = 1;
        if (2 * lo > m) lo = m / 2;
        selected.reserve(6 * lo);
        for (usize k = 0; k < lo; ++k) {
            const usize row = idx[k];
            selected.push_back(rows[3 * row + 0]);
            selected.push_back(rows[3 * row + 1]);
            selected.push_back(rows[3 * row + 2]);
        }
        for (usize k = m - lo; k < m; ++k) {
            const usize row = idx[k];
            selected.push_back(rows[3 * row + 0]);
            selected.push_back(rows[3 * row + 1]);
            selected.push_back(rows[3 * row + 2]);
        }
    }
    const usize m_sel = selected.size() / 3;

    auto set_mean_selected = [&]() noexcept {
        f64 sr = 0.0;
        f64 sg = 0.0;
        f64 sb = 0.0;
        for (usize k = 0; k < m_sel; ++k) {
            sr += selected[3 * k + 0];
            sg += selected[3 * k + 1];
            sb += selected[3 * k + 2];
        }
        const f64 inv = 1.0 / as<f64>(m_sel);
        out = Color3{sr * inv, sg * inv, sb * inv};
    };

    if (m_sel < 2) {
        set_mean_selected();
        return ImageStatus::kOk;
    }

    Sym3 cov{0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
    for (usize k = 0; k < m_sel; ++k) {
        const f64 cr = selected[3 * k + 0];
        const f64 cg = selected[3 * k + 1];
        const f64 cb = selected[3 * k + 2];
        cov.m00 += cr * cr;
        cov.m01 += cr * cg;
        cov.m02 += cr * cb;
        cov.m11 += cg * cg;
        cov.m12 += cg * cb;
        cov.m22 += cb * cb;
    }

    f64 evals[3]{};
    f64 evecs[9]{};
    jacobi_eigen_3x3(cov, evals, evecs);

    usize order[3] = {0, 1, 2};
    if (evals[order[0]] < evals[order[1]]) keel::swap(order[0], order[1]);
    if (evals[order[1]] < evals[order[2]]) keel::swap(order[1], order[2]);
    if (evals[order[0]] < evals[order[1]]) keel::swap(order[0], order[1]);

    bool v_is_identity = true;
    for (i32 c = 0; c < 3; ++c) {
        i32 max_row = 0;
        f64 max_abs = abs(evecs[0 + 3 * order[c]]);
        for (i32 row = 1; row < 3; ++row) {
            const f64 v = abs(evecs[row + 3 * order[c]]);
            if (v > max_abs) {
                max_abs = v;
                max_row = row;
            }
        }
        if (max_row != c || max_abs < 0.9999) {
            v_is_identity = false;
            break;
        }
    }
    if (v_is_identity) {
        set_mean_selected();
        return ImageStatus::kOk;
    }

    const f64 s0 = sqrt(max(0.0, evals[order[0]]));
    const f64 s1 = sqrt(max(0.0, evals[order[1]]));
    const f64 s2 = sqrt(max(0.0, evals[order[2]]));
    const f64 eps10 = 10.0 * Limits<f64>::kEpsilon;
    if ((s0 - s1) <= eps10 && (s0 - s2) <= eps10) {
        set_mean_selected();
        return ImageStatus::kOk;
    }

    const usize col = order[0];
    out = Color3{
        abs(evecs[0 + 3 * col]),
        abs(evecs[1 + 3 * col]),
        abs(evecs[2 + 3 * col]),
    };
    return ImageStatus::kOk;
}

f64 delta_e_94(Color3 lab1, Color3 lab2, f64 kL, f64 kC, f64 kH, f64 K1, f64 K2) noexcept {
    const f64 dL = lab1.c0 - lab2.c0;
    const f64 c1s = sqrt(lab1.c1 * lab1.c1 + lab1.c2 * lab1.c2);
    const f64 c2s = sqrt(lab2.c1 * lab2.c1 + lab2.c2 * lab2.c2);
    const f64 dCab = c1s - c2s;
    const f64 da = lab1.c1 - lab2.c1;
    const f64 db = lab1.c2 - lab2.c2;
    const f64 dHab_sq = da * da + db * db - dCab * dCab;
    const f64 sL = 1.0;
    const f64 sC = 1.0 + K1 * c1s;
    const f64 sH = 1.0 + K2 * c1s;
    const f64 tL = dL / (kL * sL);
    const f64 tC = dCab / (kC * sC);
    const f64 t_inside = tL * tL + tC * tC + dHab_sq / ((kH * sH) * (kH * sH));
    return sqrt(max(0.0, t_inside));
}

f64 delta_e_2000(Color3 lab1, Color3 lab2, f64 kL, f64 kC, f64 kH, f64 K1, f64 K2) noexcept {
    constexpr f64 kDeg2Rad = 0.017453292519943295769;
    const f64 l1 = lab1.c0;
    const f64 a1_in = lab1.c1;
    const f64 b1 = lab1.c2;
    const f64 l2 = lab2.c0;
    const f64 a2_in = lab2.c1;
    const f64 b2 = lab2.c2;

    const f64 c1 = sqrt(a1_in * a1_in + b1 * b1);
    const f64 c2 = sqrt(a2_in * a2_in + b2 * b2);
    const f64 cbar = 0.5 * (c1 + c2);
    const f64 cbar7 = pow(cbar, 7.0);
    const f64 g_factor = 0.5 * (1.0 - sqrt(cbar7 / (cbar7 + 6103515625.0)));
    const f64 a1 = (1.0 + g_factor) * a1_in;
    const f64 a2 = (1.0 + g_factor) * a2_in;
    const f64 c1d = sqrt(a1 * a1 + b1 * b1);
    const f64 c2d = sqrt(a2 * a2 + b2 * b2);

    const f64 h1 = wrap_hue_rad(a1, b1);
    const f64 h2 = wrap_hue_rad(a2, b2);

    const f64 dL = l2 - l1;
    const f64 dC = c2d - c1d;
    f64 dh = 0.0;
    if (c1d * c2d != 0.0) {
        const f64 hsub = h2 - h1;
        if (hsub > kPi) dh = hsub - kTau;
        else if (hsub < -kPi) dh = hsub + kTau;
        else dh = hsub;
    }
    const f64 dH = 2.0 * sqrt(c1d * c2d) * sin(dh / 2.0);

    const f64 lbar = 0.5 * (l1 + l2);
    const f64 cdbar = 0.5 * (c1d + c2d);
    f64 hbar = 0.0;
    if (c1d * c2d == 0.0) {
        hbar = h1 + h2;
    } else {
        const f64 hsub = h2 - h1;
        const f64 hadd = h1 + h2;
        if (abs(hsub) <= kPi) hbar = hadd / 2.0;
        else if (hadd < kTau) hbar = (hadd + kTau) / 2.0;
        else hbar = (hadd - kTau) / 2.0;
    }

    const f64 t_factor = 1.0 - 0.17 * cos(hbar - 30.0 * kDeg2Rad) +
                         0.24 * cos(2.0 * hbar) +
                         0.32 * cos(3.0 * hbar + 6.0 * kDeg2Rad) -
                         0.20 * cos(4.0 * hbar - 63.0 * kDeg2Rad);
    const f64 dtheta_arg = (hbar - 275.0 * kDeg2Rad) / (25.0 * kDeg2Rad);
    const f64 dtheta = 30.0 * kDeg2Rad * exp(-dtheta_arg * dtheta_arg);
    const f64 cdbar7 = pow(cdbar, 7.0);
    const f64 rC = 2.0 * sqrt(cdbar7 / (cdbar7 + 6103515625.0));
    const f64 lbar_d = lbar - 50.0;
    const f64 sL = 1.0 + (K2 * lbar_d * lbar_d) / sqrt(20.0 + lbar_d * lbar_d);
    const f64 sC = 1.0 + K1 * cdbar;
    const f64 sH = 1.0 + K2 * cdbar * t_factor;
    const f64 rT = -sin(2.0 * dtheta) * rC;

    const f64 tL = dL / (kL * sL);
    const f64 tC = dC / (kC * sC);
    const f64 tH = dH / (kH * sH);
    const f64 inside = tL * tL + tC * tC + tH * tH + rT * tC * tH;
    return sqrt(max(0.0, inside));
}

ImageStatus colordiff_lab_planar(ArrayRef<f64> l1, ArrayRef<f64> a1, ArrayRef<f64> b1,
                                 ArrayRef<f64> l2, ArrayRef<f64> a2, ArrayRef<f64> b2,
                                 ColorDiffStandard standard, f64 kL, f64 kC, f64 kH,
                                 f64 K1, f64 K2, MutArrayRef<f64> out) noexcept {
    const usize n = l1.size();
    if (a1.size() != n || b1.size() != n || l2.size() != n || a2.size() != n ||
        b2.size() != n || out.size() != n) {
        return ImageStatus::kInvalidArgument;
    }
    if (!(kL > 0.0) || !(kC > 0.0) || !(kH > 0.0) || !(K1 > 0.0) || !(K2 > 0.0)) {
        return ImageStatus::kInvalidArgument;
    }
    if (standard == ColorDiffStandard::kCiede2000) {
        for (usize i = 0; i < n; ++i) {
            out[i] = delta_e_2000(Color3{l1[i], a1[i], b1[i]},
                                  Color3{l2[i], a2[i], b2[i]}, kL, kC, kH, K1, K2);
        }
    } else {
        for (usize i = 0; i < n; ++i) {
            out[i] = delta_e_94(Color3{l1[i], a1[i], b1[i]},
                                Color3{l2[i], a2[i], b2[i]}, kL, kC, kH, K1, K2);
        }
    }
    return ImageStatus::kOk;
}

} // namespace keel::image
