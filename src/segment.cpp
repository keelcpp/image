#include "keel/image.h"

#include "keel/algo.h"
#include "keel/math.h"
#include "keel/static_array.h"

namespace keel::image {

namespace {

template <typename T, typename Cmp>
class MinHeap {
public:
    explicit MinHeap(Cmp cmp, Allocator* a) : m_data(a), m_cmp(cmp) {}
    bool empty() const noexcept { return m_data.empty(); }
    const T& top() const noexcept { return m_data.front(); }
    void push(const T& v) {
        m_data.push_back(v);
        push_heap(m_data.begin(), m_data.end(), m_cmp);
    }
    void pop() {
        pop_heap(m_data.begin(), m_data.end(), m_cmp);
        m_data.pop_back();
    }

private:
    Array<T> m_data;
    Cmp m_cmp;
};

struct ChamferStep {
    i32 dr = 0;
    i32 dc = 0;
    f64 w = 0.0;
};

i32 fill_chamfer_steps(GeodesicMetric metric, StaticArray<ChamferStep, 8>& steps) noexcept {
    if (metric == GeodesicMetric::kCityblock) {
        steps[0] = {-1, 0, 1.0};
        steps[1] = {1, 0, 1.0};
        steps[2] = {0, -1, 1.0};
        steps[3] = {0, 1, 1.0};
        return 4;
    }
    const f64 diag = (metric == GeodesicMetric::kChessboard) ? 1.0 : sqrt(2.0);
    steps[0] = {-1, 0, 1.0};
    steps[1] = {1, 0, 1.0};
    steps[2] = {0, -1, 1.0};
    steps[3] = {0, 1, 1.0};
    steps[4] = {-1, -1, diag};
    steps[5] = {-1, 1, diag};
    steps[6] = {1, -1, diag};
    steps[7] = {1, 1, diag};
    return 8;
}

template <typename T>
ImageStatus reducepoly_impl(ArrayRef<T> xs, ArrayRef<T> ys, f64 tolerance,
                            Array<i32>& out_kept_indices) {
    if (xs.size() != ys.size() || !is_finite(tolerance) || !(tolerance >= 0.0 && tolerance <= 1.0)) {
        return ImageStatus::kInvalidArgument;
    }
    out_kept_indices.clear();
    const usize n = xs.size();
    if (n == 0) {
        return ImageStatus::kOk;
    }
    if (n == 1) {
        out_kept_indices.push_back(0);
        return ImageStatus::kOk;
    }
    T min_x = xs[0];
    T max_x = xs[0];
    T min_y = ys[0];
    T max_y = ys[0];
    for (usize i = 1; i < n; ++i) {
        if (xs[i] < min_x) min_x = xs[i];
        if (xs[i] > max_x) max_x = xs[i];
        if (ys[i] < min_y) min_y = ys[i];
        if (ys[i] > max_y) max_y = ys[i];
    }
    const T sdx = max_x - min_x;
    const T sdy = max_y - min_y;
    const T scaling = sqrt(sdx * sdx + sdy * sdy);
    constexpr f64 kEps = 2.220446049250313e-16;
    const T tol_eff = as<T>((tolerance == 0.0) ? kEps : tolerance) * scaling;

    // Iterative Ramer-Douglas-Peucker to avoid deep recursion stack overflow.
    Array<u8> keep(n, u8{0});
    keep[0] = 1;
    keep[n - 1] = 1;
    struct Span {
        i32 lo;
        i32 hi;
    };
    Array<Span> stack;
    stack.reserve(64);
    stack.push_back(Span{0, as<i32>(n - 1)});
    while (!stack.empty()) {
        const Span sp = stack.back();
        stack.pop_back();
        if (sp.hi - sp.lo <= 1) {
            continue;
        }
        const T x1 = xs[as<usize>(sp.lo)];
        const T y1 = ys[as<usize>(sp.lo)];
        const T x2 = xs[as<usize>(sp.hi)];
        const T y2 = ys[as<usize>(sp.hi)];
        const T dx = x2 - x1;
        const T dy = y2 - y1;
        const T d_node = sqrt(dx * dx + dy * dy);

        T max_d = as<T>(-1);
        i32 far_idx = -1;
        for (i32 g = sp.lo + 1; g < sp.hi; ++g) {
            const T xk = xs[as<usize>(g)];
            const T yk = ys[as<usize>(g)];
            T d = as<T>(0);
            if (as<f64>(d_node) > kEps) {
                const T det = (x2 * yk - y2 * xk) - x1 * (yk - y2) + y1 * (xk - x2);
                d = abs(det) / d_node;
            } else {
                const T ex = xk - x1;
                const T ey = yk - y1;
                d = sqrt(ex * ex + ey * ey);
            }
            if (d > max_d) {
                max_d = d;
                far_idx = g;
            }
        }
        if (max_d > tol_eff && far_idx > sp.lo && far_idx < sp.hi) {
            keep[as<usize>(far_idx)] = 1;
            stack.push_back(Span{far_idx, sp.hi});
            stack.push_back(Span{sp.lo, far_idx});
        }
    }
    for (usize i = 0; i < n; ++i) {
        if (keep[i] != 0) {
            out_kept_indices.push_back(as<i32>(i));
        }
    }
    return ImageStatus::kOk;
}

} // namespace

f64 dice_coeff(ArrayRef<u8> a, ArrayRef<u8> b) noexcept {
    if (a.size() != b.size()) {
        return kNaN;
    }
    i64 inter = 0;
    i64 sum_a = 0;
    i64 sum_b = 0;
    for (usize i = 0; i < a.size(); ++i) {
        const bool av = (a[i] != 0);
        const bool bv = (b[i] != 0);
        if (av) ++sum_a;
        if (bv) ++sum_b;
        if (av && bv) ++inter;
    }
    const i64 denom = sum_a + sum_b;
    return (denom == 0) ? 1.0 : (2.0 * as<f64>(inter) / as<f64>(denom));
}

f64 jaccard_index(ArrayRef<u8> a, ArrayRef<u8> b) noexcept {
    if (a.size() != b.size()) {
        return kNaN;
    }
    i64 inter = 0;
    i64 uni = 0;
    for (usize i = 0; i < a.size(); ++i) {
        const bool av = (a[i] != 0);
        const bool bv = (b[i] != 0);
        if (av && bv) ++inter;
        if (av || bv) ++uni;
    }
    return (uni == 0) ? 1.0 : (as<f64>(inter) / as<f64>(uni));
}

ImageStatus boundarymask_2d(ArrayRef<i64> labels_col_major, usize height, usize width, i32 conn,
                            MutArrayRef<u8> out_col_major) noexcept {
    const usize total = height * width;
    if (labels_col_major.size() != total || out_col_major.size() != total) {
        return ImageStatus::kInvalidArgument;
    }
    if (conn != 4) {
        conn = 8;
    }
    if (total == 0) {
        return ImageStatus::kOk;
    }
    static constexpr i32 kDr8[8] = {-1, -1, -1, 0, 0, 1, 1, 1};
    static constexpr i32 kDc8[8] = {-1, 0, 1, -1, 1, -1, 0, 1};
    static constexpr i32 kDr4[4] = {-1, 1, 0, 0};
    static constexpr i32 kDc4[4] = {0, 0, -1, 1};
    const i32* dr = (conn == 4) ? kDr4 : kDr8;
    const i32* dc = (conn == 4) ? kDc4 : kDc8;
    const i32 k_cnt = (conn == 4) ? 4 : 8;
    const i32 h = as<i32>(height);
    const i32 w = as<i32>(width);

    for (i32 c = 0; c < w; ++c) {
        for (i32 r = 0; r < h; ++r) {
            const usize idx = as<usize>(c) * height + as<usize>(r);
            const i64 me = labels_col_major[idx];
            if (me == 0) {
                out_col_major[idx] = 0;
                continue;
            }
            bool is_bnd = false;
            for (i32 k = 0; k < k_cnt; ++k) {
                const i32 nr = r + dr[k];
                const i32 nc = c + dc[k];
                if (nr < 0 || nr >= h || nc < 0 || nc >= w) {
                    is_bnd = true;
                    break;
                }
                if (labels_col_major[as<usize>(nc) * height + as<usize>(nr)] != me) {
                    is_bnd = true;
                    break;
                }
            }
            out_col_major[idx] = is_bnd ? u8{1} : u8{0};
        }
    }
    return ImageStatus::kOk;
}

ImageStatus grayconnected_2d(ArrayRef<f64> img_col_major, usize height, usize width, i32 seed_row,
                             i32 seed_col, f64 tol, MutArrayRef<u8> out_col_major,
                             Allocator* alloc) {
    const usize total = height * width;
    if (img_col_major.size() != total || out_col_major.size() != total || is_nan(tol) ||
        tol < 0.0) {
        return ImageStatus::kInvalidArgument;
    }
    fill(out_col_major.begin(), out_col_major.end(), u8{0});
    if (total == 0) {
        return ImageStatus::kOk;
    }
    const i32 h = as<i32>(height);
    const i32 w = as<i32>(width);
    if (seed_row < 0 || seed_row >= h || seed_col < 0 || seed_col >= w) {
        return ImageStatus::kInvalidArgument;
    }

    Allocator* a = alloc ? alloc : default_allocator();
    const usize seed_idx = as<usize>(seed_col) * height + as<usize>(seed_row);
    const f64 seed_val = img_col_major[seed_idx];
    out_col_major[seed_idx] = 1;

    Array<usize> q(a);
    q.reserve(total / 4 + 4);
    q.push_back(seed_idx);

    static constexpr i32 kDr8[8] = {-1, -1, -1, 0, 0, 1, 1, 1};
    static constexpr i32 kDc8[8] = {-1, 0, 1, -1, 1, -1, 0, 1};
    usize head = 0;
    while (head < q.size()) {
        const usize curr = q[head++];
        const i32 r = as<i32>(curr % height);
        const i32 c = as<i32>(curr / height);
        for (i32 k = 0; k < 8; ++k) {
            const i32 nr = r + kDr8[k];
            const i32 nc = c + kDc8[k];
            if (nr < 0 || nr >= h || nc < 0 || nc >= w) {
                continue;
            }
            const usize nb = as<usize>(nc) * height + as<usize>(nr);
            if (out_col_major[nb]) {
                continue;
            }
            if (!(abs(img_col_major[nb] - seed_val) <= tol)) {
                continue;
            }
            out_col_major[nb] = 1;
            q.push_back(nb);
        }
    }
    return ImageStatus::kOk;
}

ImageStatus graydiffweight_f64(ArrayRef<f64> img, f64 ref_gray_val, f64 rolloff_factor, f64 cutoff,
                               MutArrayRef<f64> out_weights) noexcept {
    if (img.size() != out_weights.size() || !is_finite(ref_gray_val) ||
        !is_finite(rolloff_factor) || !(rolloff_factor > 0.0) || !(cutoff >= 0.0)) {
        return ImageStatus::kInvalidArgument;
    }
    const usize n = img.size();
    if (n == 0) {
        return ImageStatus::kOk;
    }
    f64 dmin = Limits<f64>::infinity();
    f64 dmax = -Limits<f64>::infinity();
    for (usize i = 0; i < n; ++i) {
        const f64 v = abs(img[i] - ref_gray_val);
        out_weights[i] = v;
        if (v < dmin) dmin = v;
        if (v > dmax) dmax = v;
    }
    constexpr f64 kLo = 1e-3;
    constexpr f64 kHi = 1.0;
    const f64 range = dmax - dmin;
    const f64 slope = (range > 0.0) ? ((kHi - kLo) / range) : 0.0;
    const f64 inv_rolloff = 1.0 / rolloff_factor;
    const bool finite_cutoff = is_finite(cutoff);
    for (usize i = 0; i < n; ++i) {
        const f64 d = out_weights[i];
        f64 s = (range > 0.0) ? (kLo + slope * (d - dmin)) : kLo;
        if (finite_cutoff && d > cutoff) {
            s = kHi;
        }
        out_weights[i] = 1.0 / pow(s, inv_rolloff);
    }
    return ImageStatus::kOk;
}

ImageStatus gradientweight_2d(ArrayRef<f64> img_col_major, usize height, usize width, f64 sigma_x,
                              f64 sigma_y, f64 rolloff_factor, f64 weight_cutoff,
                              MutArrayRef<f64> out_weights_col_major, Allocator* alloc) {
    const usize total = height * width;
    if (img_col_major.size() != total || out_weights_col_major.size() != total
        || !(sigma_x > 0.0) || !is_finite(sigma_x) || !(sigma_y > 0.0) || !is_finite(sigma_y)
        || !(rolloff_factor > 0.0) || !is_finite(rolloff_factor)
        || !(weight_cutoff >= 1e-3 && weight_cutoff <= 1.0)) {
        return ImageStatus::kInvalidArgument;
    }
    if (total == 0) {
        return ImageStatus::kOk;
    }

    f64 min_i = Limits<f64>::infinity();
    f64 max_i = -Limits<f64>::infinity();
    for (usize i = 0; i < total; ++i) {
        if (img_col_major[i] < min_i) min_i = img_col_major[i];
        if (img_col_major[i] > max_i) max_i = img_col_major[i];
    }
    const f64 abs_max = max(abs(min_i), abs(max_i));
    const f64 eps_i = nextafter(abs_max, Limits<f64>::infinity()) - abs_max;
    if (max_i - eps_i * 1000.0 <= min_i) {
        fill(out_weights_col_major.begin(), out_weights_col_major.end(), 1.0);
        return ImageStatus::kOk;
    }

    Allocator* a = alloc ? alloc : default_allocator();
    auto build_dog = [&](f64 sigma, i32 norm_count) -> Array<f64> {
        const i32 r = as<i32>(ceil(2.0 * sigma));
        const usize len = as<usize>(2 * r + 1);
        Array<f64> h(a);
        h.resize(len);
        f64 norm = 0.0;
        for (i32 k = 0; k < as<i32>(len); ++k) {
            const i32 x = k - r;
            h[as<usize>(k)] =
                -as<f64>(x) * exp(-as<f64>(x * x) / (2.0 * sigma * sigma));
            if (k < norm_count) {
                norm += h[as<usize>(k)];
            }
        }
        if (norm == 0.0) {
            norm = 1.0;
        }
        for (usize k = 0; k < len; ++k) {
            h[k] /= norm;
        }
        return h;
    };

    const i32 rx = as<i32>(ceil(2.0 * sigma_x));
    const i32 ry = as<i32>(ceil(2.0 * sigma_y));
    Array<f64> hx = build_dog(sigma_x, rx);
    Array<f64> hy = build_dog(sigma_y, ry);

    Array<f64> gx(a);
    gx.resize(total);
    Array<f64> gy(a);
    gy.resize(total);
    (void)imfilter_2d(img_col_major, height, width, hx.as_ref(), 1, hx.size(), PadMode::kReplicate,
                      0.0, false, false, gx.as_mut_ref());
    (void)imfilter_2d(img_col_major, height, width, hy.as_ref(), hy.size(), 1, PadMode::kReplicate,
                      0.0, false, false, gy.as_mut_ref());

    Array<f64> gmag(a);
    gmag.resize(total);
    f64 gmin = Limits<f64>::infinity();
    f64 gmax = -Limits<f64>::infinity();
    for (usize i = 0; i < total; ++i) {
        const f64 g = hypot(gx[i], gy[i]);
        gmag[i] = g;
        if (g < gmin) gmin = g;
        if (g > gmax) gmax = g;
    }
    const f64 gabs_min = min(abs(gmin), abs(gmax));
    const f64 eps_g = nextafter(gabs_min, Limits<f64>::infinity()) - gabs_min;
    const bool spread_ok = (gmax - gmin) > eps_g;
    const f64 slope = spread_ok ? (1.0 / (gmax - gmin)) : 0.0;
    const f64 inv_rolloff = 1.0 / rolloff_factor;

    for (usize i = 0; i < total; ++i) {
        f64 w = spread_ok ? (slope * (gmag[i] - gmin)) : 0.0;
        w = pow(w, inv_rolloff);
        w = (1.0 - w) / (1.0 + w);
        if (w < weight_cutoff) {
            w = 1e-3;
        }
        out_weights_col_major[i] = w;
    }
    return ImageStatus::kOk;
}

ImageStatus regionfill_2d(ArrayRef<f64> img_col_major, ArrayRef<u8> mask_col_major, usize height,
                          usize width, MutArrayRef<f64> out_col_major, Allocator* alloc) {
    const usize total = height * width;
    if (height < 3 || width < 3 || img_col_major.size() != total || mask_col_major.size() != total
        || out_col_major.size() != total) {
        return ImageStatus::kInvalidArgument;
    }
    copy_n(out_col_major.data(), img_col_major.data(), total);

    Allocator* a = alloc ? alloc : default_allocator();
    auto lin = [&](usize r, usize c) noexcept -> usize { return c * height + r; };

    Array<i32> idx_of(a);
    idx_of.assign(total, i32{-1});
    Array<usize> pix_of(a);
    pix_of.reserve(total / 4 + 4);
    for (usize i = 0; i < total; ++i) {
        if (mask_col_major[i]) {
            idx_of[i] = as<i32>(pix_of.size());
            pix_of.push_back(i);
        }
    }
    const usize nu = pix_of.size();
    if (nu == 0) {
        return ImageStatus::kOk;
    }

    Array<i32> num_nbrs(a);
    num_nbrs.assign(nu, i32{0});
    Array<f64> rhs(a);
    rhs.assign(nu, 0.0);
    Array<i32> nbrs(a);
    nbrs.assign(nu * 4, i32{-1});

    auto add_dir = [&](usize i, usize nr, usize nc, bool valid, usize slot) noexcept {
        if (!valid) {
            return;
        }
        const usize nk = lin(nr, nc);
        ++num_nbrs[i];
        if (mask_col_major[nk]) {
            nbrs[i * 4 + slot] = idx_of[nk];
        } else {
            rhs[i] += img_col_major[nk];
        }
    };

    for (usize i = 0; i < nu; ++i) {
        const usize k = pix_of[i];
        const usize c = k / height;
        const usize r = k % height;
        add_dir(i, r - 1, c, r > 0, 0);
        add_dir(i, r + 1, c, r + 1 < height, 1);
        add_dir(i, r, c - 1, c > 0, 2);
        add_dir(i, r, c + 1, c + 1 < width, 3);
    }

    auto matvec = [&](ArrayRef<f64> x, MutArrayRef<f64> y) noexcept {
        for (usize i = 0; i < nu; ++i) {
            f64 s = as<f64>(num_nbrs[i]) * x[i];
            const i32* nb = nbrs.data() + i * 4;
            if (nb[0] >= 0) s -= x[as<usize>(nb[0])];
            if (nb[1] >= 0) s -= x[as<usize>(nb[1])];
            if (nb[2] >= 0) s -= x[as<usize>(nb[2])];
            if (nb[3] >= 0) s -= x[as<usize>(nb[3])];
            y[i] = s;
        }
    };

    Array<f64> x(a);
    x.assign(nu, 0.0);
    Array<f64> r(a);
    r.resize(nu);
    Array<f64> p(a);
    p.resize(nu);
    Array<f64> ap(a);
    ap.resize(nu);

    f64 rdotr = 0.0;
    f64 bnorm = 0.0;
    for (usize i = 0; i < nu; ++i) {
        r[i] = rhs[i];
        p[i] = rhs[i];
        rdotr += r[i] * r[i];
        bnorm += rhs[i] * rhs[i];
    }
    const f64 bn = sqrt(bnorm);
    const f64 tol = (bn > 0.0) ? ((1e-12 * bn) * (1e-12 * bn)) : 1e-24;

    const usize max_iter = nu * 4 + 200;
    for (usize it = 0; it < max_iter; ++it) {
        if (!is_finite(rdotr) || rdotr <= tol) {
            break;
        }
        matvec(p.as_ref(), ap.as_mut_ref());
        f64 pap = 0.0;
        for (usize i = 0; i < nu; ++i) {
            pap += p[i] * ap[i];
        }
        if (!is_finite(pap) || pap <= 0.0) {
            break;
        }
        const f64 alpha = rdotr / pap;
        for (usize i = 0; i < nu; ++i) {
            x[i] += alpha * p[i];
            r[i] -= alpha * ap[i];
        }
        f64 rdotr_new = 0.0;
        for (usize i = 0; i < nu; ++i) {
            rdotr_new += r[i] * r[i];
        }
        const f64 beta = rdotr_new / rdotr;
        for (usize i = 0; i < nu; ++i) {
            p[i] = r[i] + beta * p[i];
        }
        rdotr = rdotr_new;
    }

    for (usize i = 0; i < nu; ++i) {
        out_col_major[pix_of[i]] = x[i];
    }
    return ImageStatus::kOk;
}

ImageStatus poly2mask_2d(ArrayRef<f64> vx_1based, ArrayRef<f64> vy_1based, usize height,
                         usize width, MutArrayRef<u8> out_bw_col_major, Allocator* alloc) {
    if (vx_1based.size() != vy_1based.size() || out_bw_col_major.size() != height * width) {
        return ImageStatus::kInvalidArgument;
    }
    fill(out_bw_col_major.begin(), out_bw_col_major.end(), u8{0});
    const usize nx = vx_1based.size();
    if (nx == 0 || height == 0 || width == 0) {
        return ImageStatus::kOk;
    }

    Allocator* a = alloc ? alloc : default_allocator();
    Array<f64> vx(a);
    Array<f64> vy(a);
    vx.reserve(nx + 1);
    vy.reserve(nx + 1);
    for (usize i = 0; i < nx; ++i) {
        vx.push_back(vx_1based[i]);
        vy.push_back(vy_1based[i]);
    }
    if (vx.front() != vx.back() || vy.front() != vy.back()) {
        vx.push_back(vx.front());
        vy.push_back(vy.front());
    }
    const usize nv = vx.size();

    for (usize c = 0; c < width; ++c) {
        const f64 cx = as<f64>(c + 1);
        const usize col_off = c * height;
        for (usize r = 0; r < height; ++r) {
            const f64 cy = as<f64>(r + 1);
            i32 crossings = 0;
            for (usize i = 0; i + 1 < nv; ++i) {
                const f64 y1 = vy[i];
                const f64 y2 = vy[i + 1];
                if (y1 == y2) {
                    continue;
                }
                const f64 ylo = min(y1, y2);
                const f64 yhi = max(y1, y2);
                if (cy <= ylo || cy > yhi) {
                    continue;
                }
                const f64 x1 = vx[i];
                const f64 x2 = vx[i + 1];
                const f64 xi = x1 + (cy - y1) * (x2 - x1) / (y2 - y1);
                if (cx > xi) {
                    crossings ^= 1;
                }
            }
            out_bw_col_major[col_off + r] = as<u8>(crossings);
        }
    }
    return ImageStatus::kOk;
}

ImageStatus graydist_2d(ArrayRef<f64> img_col_major, usize height, usize width,
                        ArrayRef<usize> seed_indices_0based, GeodesicMetric metric,
                        MutArrayRef<f64> out_dist_col_major, Allocator* alloc) {
    const usize total = height * width;
    if (img_col_major.size() != total || out_dist_col_major.size() != total) {
        return ImageStatus::kInvalidArgument;
    }
    for (usize i = 0; i < total; ++i) {
        if (img_col_major[i] < 0.0) {
            return ImageStatus::kInvalidArgument;
        }
    }
    const f64 inf = Limits<f64>::infinity();
    fill(out_dist_col_major.begin(), out_dist_col_major.end(), inf);
    if (total == 0 || seed_indices_0based.size() == 0) {
        return ImageStatus::kOk;
    }

    Allocator* a = alloc ? alloc : default_allocator();
    struct Node {
        f64 dist;
        usize idx;
    };
    auto cmp = [](const Node& x, const Node& y) noexcept { return x.dist > y.dist; };
    MinHeap<Node, decltype(cmp)> pq(cmp, a);

    for (usize k = 0; k < seed_indices_0based.size(); ++k) {
        const usize idx = seed_indices_0based[k];
        if (idx >= total) {
            return ImageStatus::kInvalidArgument;
        }
        out_dist_col_major[idx] = 0.0;
        pq.push(Node{0.0, idx});
    }

    StaticArray<ChamferStep, 8> steps{};
    const i32 n_steps = fill_chamfer_steps(metric, steps);
    while (!pq.empty()) {
        const Node top = pq.top();
        pq.pop();
        if (top.dist > out_dist_col_major[top.idx]) {
            continue;
        }
        const usize r = top.idx % height;
        const usize c = top.idx / height;
        const f64 iq = img_col_major[top.idx];
        for (i32 s = 0; s < n_steps; ++s) {
            const i64 nr = as<i64>(r) + steps[as<usize>(s)].dr;
            const i64 nc = as<i64>(c) + steps[as<usize>(s)].dc;
            if (nr < 0 || nc < 0 || as<usize>(nr) >= height || as<usize>(nc) >= width) {
                continue;
            }
            const usize nk = as<usize>(nc) * height + as<usize>(nr);
            const f64 nd = top.dist + steps[as<usize>(s)].w * (iq + img_col_major[nk]) * 0.5;
            if (nd < out_dist_col_major[nk]) {
                out_dist_col_major[nk] = nd;
                pq.push(Node{nd, nk});
            }
        }
    }
    return ImageStatus::kOk;
}

ImageStatus bwdistgeodesic_2d(ArrayRef<u8> bw_col_major, usize height, usize width,
                              ArrayRef<usize> seed_indices_0based, GeodesicMetric metric,
                              MutArrayRef<f64> out_dist_col_major, Allocator* alloc) {
    const usize total = height * width;
    if (bw_col_major.size() != total || out_dist_col_major.size() != total) {
        return ImageStatus::kInvalidArgument;
    }
    const f64 inf = Limits<f64>::infinity();
    for (usize i = 0; i < total; ++i) {
        out_dist_col_major[i] = bw_col_major[i] ? inf : kNaN;
    }
    if (total == 0 || seed_indices_0based.size() == 0) {
        return ImageStatus::kOk;
    }

    Allocator* a = alloc ? alloc : default_allocator();
    struct Node {
        f64 dist;
        usize idx;
    };
    auto cmp = [](const Node& x, const Node& y) noexcept { return x.dist > y.dist; };
    MinHeap<Node, decltype(cmp)> pq(cmp, a);

    for (usize k = 0; k < seed_indices_0based.size(); ++k) {
        const usize idx = seed_indices_0based[k];
        if (idx >= total) {
            return ImageStatus::kInvalidArgument;
        }
        if (!bw_col_major[idx]) {
            continue;
        }
        out_dist_col_major[idx] = 0.0;
        pq.push(Node{0.0, idx});
    }

    StaticArray<ChamferStep, 8> steps{};
    const i32 n_steps = fill_chamfer_steps(metric, steps);
    while (!pq.empty()) {
        const Node top = pq.top();
        pq.pop();
        if (top.dist > out_dist_col_major[top.idx]) {
            continue;
        }
        const usize r = top.idx % height;
        const usize c = top.idx / height;
        for (i32 s = 0; s < n_steps; ++s) {
            const i64 nr = as<i64>(r) + steps[as<usize>(s)].dr;
            const i64 nc = as<i64>(c) + steps[as<usize>(s)].dc;
            if (nr < 0 || nc < 0 || as<usize>(nr) >= height || as<usize>(nc) >= width) {
                continue;
            }
            const usize nk = as<usize>(nc) * height + as<usize>(nr);
            if (!bw_col_major[nk]) {
                continue;
            }
            const f64 nd = top.dist + steps[as<usize>(s)].w;
            if (nd < out_dist_col_major[nk]) {
                out_dist_col_major[nk] = nd;
                pq.push(Node{nd, nk});
            }
        }
    }
    return ImageStatus::kOk;
}

ImageStatus reducepoly_f64(ArrayRef<f64> xs, ArrayRef<f64> ys, f64 tolerance,
                           Array<i32>& out_kept_indices) {
    return reducepoly_impl<f64>(xs, ys, tolerance, out_kept_indices);
}

ImageStatus reducepoly_f32(ArrayRef<f32> xs, ArrayRef<f32> ys, f64 tolerance,
                           Array<i32>& out_kept_indices) {
    return reducepoly_impl<f32>(xs, ys, tolerance, out_kept_indices);
}

ImageStatus watershed_2d(ArrayRef<f64> img_col_major, usize height, usize width, i32 conn,
                         MutArrayRef<u32> out_labels_col_major, u32& num_basins_out,
                         Allocator* alloc) {
    const usize total = height * width;
    if (img_col_major.size() != total || out_labels_col_major.size() != total
        || (conn != 4 && conn != 8)) {
        return ImageStatus::kInvalidArgument;
    }
    fill(out_labels_col_major.begin(), out_labels_col_major.end(), u32{0});
    num_basins_out = 0;
    if (total == 0) {
        return ImageStatus::kOk;
    }
    for (usize i = 0; i < total; ++i) {
        if (is_nan(img_col_major[i])) {
            return ImageStatus::kInvalidArgument;
        }
    }

    Allocator* a = alloc ? alloc : default_allocator();
    static constexpr i32 kDr8[8] = {-1, 0, 1, -1, 1, -1, 0, 1};
    static constexpr i32 kDc8[8] = {-1, -1, -1, 0, 0, 1, 1, 1};
    static constexpr i32 kDr4[4] = {-1, 1, 0, 0};
    static constexpr i32 kDc4[4] = {0, 0, -1, 1};
    const i32* dr = (conn == 4) ? kDr4 : kDr8;
    const i32* dc = (conn == 4) ? kDc4 : kDc8;
    const i32 k_cnt = (conn == 4) ? 4 : 8;
    const i32 hi = as<i32>(height);
    const i32 wi = as<i32>(width);

    // Step 1: Identify regional minima plateaus under conn-connectivity.
    Array<u8> visited(a);
    visited.assign(total, u8{0});
    Array<u8> regmin(a);
    regmin.assign(total, u8{0});
    Array<usize> comp(a);
    Array<usize> q(a);
    comp.reserve(64);
    q.reserve(64);

    for (usize i = 0; i < total; ++i) {
        if (visited[i]) {
            continue;
        }
        visited[i] = 1;
        comp.clear();
        q.clear();
        comp.push_back(i);
        q.push_back(i);
        const f64 v0 = img_col_major[i];
        bool has_lower = false;
        usize head = 0;
        while (head < q.size()) {
            const usize curr = q[head++];
            const i32 r = as<i32>(curr % height);
            const i32 c = as<i32>(curr / height);
            for (i32 k = 0; k < k_cnt; ++k) {
                const i32 nr = r + dr[k];
                const i32 nc = c + dc[k];
                if (nr < 0 || nr >= hi || nc < 0 || nc >= wi) {
                    continue;
                }
                const usize nb = as<usize>(nc) * height + as<usize>(nr);
                if (img_col_major[nb] < v0) {
                    has_lower = true;
                } else if (img_col_major[nb] == v0 && !visited[nb]) {
                    visited[nb] = 1;
                    comp.push_back(nb);
                    q.push_back(nb);
                }
            }
        }
        if (!has_lower) {
            for (usize j = 0; j < comp.size(); ++j) {
                regmin[comp[j]] = 1;
            }
        }
    }

    // Step 2: Label connected components of regmin in column-major order.
    u32 num_basins = 0;
    for (usize i = 0; i < total; ++i) {
        if (regmin[i] && out_labels_col_major[i] == 0) {
            ++num_basins;
            out_labels_col_major[i] = num_basins;
            q.clear();
            q.push_back(i);
            usize head = 0;
            while (head < q.size()) {
                const usize curr = q[head++];
                const i32 r = as<i32>(curr % height);
                const i32 c = as<i32>(curr / height);
                for (i32 k = 0; k < k_cnt; ++k) {
                    const i32 nr = r + dr[k];
                    const i32 nc = c + dc[k];
                    if (nr < 0 || nr >= hi || nc < 0 || nc >= wi) {
                        continue;
                    }
                    const usize nb = as<usize>(nc) * height + as<usize>(nr);
                    if (regmin[nb] && out_labels_col_major[nb] == 0) {
                        out_labels_col_major[nb] = num_basins;
                        q.push_back(nb);
                    }
                }
            }
        }
    }

    // Step 3: Meyer's priority queue flood.
    struct PqEntry {
        f64 val;
        usize ord;
        usize idx;
    };
    auto cmp = [](const PqEntry& x, const PqEntry& y) noexcept {
        if (x.val != y.val) return x.val > y.val;
        return x.ord > y.ord;
    };
    MinHeap<PqEntry, decltype(cmp)> pq(cmp, a);
    Array<u8> in_pq(a);
    in_pq.assign(total, u8{0});
    for (usize i = 0; i < total; ++i) {
        if (out_labels_col_major[i] > 0) {
            in_pq[i] = 1;
        }
    }
    usize ord = 0;
    for (usize i = 0; i < total; ++i) {
        if (out_labels_col_major[i] > 0) {
            const i32 r = as<i32>(i % height);
            const i32 c = as<i32>(i / height);
            for (i32 k = 0; k < k_cnt; ++k) {
                const i32 nr = r + dr[k];
                const i32 nc = c + dc[k];
                if (nr < 0 || nr >= hi || nc < 0 || nc >= wi) {
                    continue;
                }
                const usize nb = as<usize>(nc) * height + as<usize>(nr);
                if (!in_pq[nb]) {
                    in_pq[nb] = 1;
                    pq.push(PqEntry{img_col_major[nb], ++ord, nb});
                }
            }
        }
    }

    while (!pq.empty()) {
        const PqEntry top = pq.top();
        pq.pop();
        const usize curr = top.idx;
        const i32 r = as<i32>(curr % height);
        const i32 c = as<i32>(curr / height);
        u32 single_label = 0;
        bool multiple = false;
        for (i32 k = 0; k < k_cnt; ++k) {
            const i32 nr = r + dr[k];
            const i32 nc = c + dc[k];
            if (nr < 0 || nr >= hi || nc < 0 || nc >= wi) {
                continue;
            }
            const u32 nl = out_labels_col_major[as<usize>(nc) * height + as<usize>(nr)];
            if (nl > 0) {
                if (single_label == 0) {
                    single_label = nl;
                } else if (single_label != nl) {
                    multiple = true;
                    break;
                }
            }
        }
        if (single_label > 0 && !multiple) {
            out_labels_col_major[curr] = single_label;
            for (i32 k = 0; k < k_cnt; ++k) {
                const i32 nr = r + dr[k];
                const i32 nc = c + dc[k];
                if (nr < 0 || nr >= hi || nc < 0 || nc >= wi) {
                    continue;
                }
                const usize nb = as<usize>(nc) * height + as<usize>(nr);
                if (!in_pq[nb]) {
                    in_pq[nb] = 1;
                    pq.push(PqEntry{img_col_major[nb], ++ord, nb});
                }
            }
        }
    }

    num_basins_out = num_basins;
    return ImageStatus::kOk;
}

} // namespace keel::image
