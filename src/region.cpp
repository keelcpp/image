#include "keel/image.h"

#include "keel/algo.h"
#include "keel/math.h"

namespace keel::image {

namespace {

// Tomas-Holst 2x2 corner pattern perimeter weights.
constexpr f64 kPerimWeights[16] = {
    0.0,      // 0000
    1.207107, // 0001
    1.207107, // 0010
    1.0,      // 0011
    1.207107, // 0100
    1.0,      // 0101
    2.414214, // 0110
    1.207107, // 0111
    1.207107, // 1000
    2.414214, // 1001
    1.0,      // 1010
    1.207107, // 1011
    1.0,      // 1100
    1.207107, // 1101
    1.207107, // 1110
    0.0       // 1111
};

void dt_1d_parabolic(ArrayRef<f64> f, MutArrayRef<f64> d, MutArrayRef<i64> v,
                     MutArrayRef<f64> z) noexcept {
    const usize n = f.size();
    if (n == 0) {
        return;
    }
    const f64 inf = Limits<f64>::infinity();
    i64 k = 0;
    v[0] = 0;
    z[0] = -inf;
    z[1] = inf;
    for (i64 q = 1; q < as<i64>(n); ++q) {
        f64 s = 0.0;
        for (;;) {
            const f64 fq = f[as<usize>(q)];
            const f64 fv = f[as<usize>(v[as<usize>(k)])];
            const f64 dq = as<f64>(q);
            const f64 dv = as<f64>(v[as<usize>(k)]);
            if (is_inf(fq) && is_inf(fv)) {
                s = -inf;
            } else {
                s = ((fq + dq * dq) - (fv + dv * dv)) / (2.0 * (dq - dv));
            }
            if (s <= z[as<usize>(k)]) {
                --k;
                if (k < 0) {
                    break;
                }
            } else {
                break;
            }
        }
        ++k;
        v[as<usize>(k)] = q;
        z[as<usize>(k)] = s;
        z[as<usize>(k) + 1] = inf;
    }
    i64 kk = 0;
    for (i64 q = 0; q < as<i64>(n); ++q) {
        while (z[as<usize>(kk) + 1] < as<f64>(q)) {
            ++kk;
        }
        const f64 dx = as<f64>(q - v[as<usize>(kk)]);
        d[as<usize>(q)] = dx * dx + f[as<usize>(v[as<usize>(kk)])];
    }
}

struct Point2D {
    f64 x = 0.0;
    f64 y = 0.0;
};

f64 cross_2d(const Point2D& o, const Point2D& a, const Point2D& b) noexcept {
    return (a.x - o.x) * (b.y - o.y) - (a.y - o.y) * (b.x - o.x);
}

f64 convex_hull_area(Array<Point2D>& pts, Allocator* a) {
    const usize n = pts.size();
    if (n < 3) {
        return 0.0;
    }
    sort(pts.begin(), pts.end(), [](const Point2D& p, const Point2D& q) noexcept {
        if (p.x != q.x) {
            return p.x < q.x;
        }
        return p.y < q.y;
    });
    Array<Point2D> hull(a);
    hull.reserve(2 * n);
    for (usize i = 0; i < n; ++i) {
        const Point2D& p = pts[i];
        while (hull.size() >= 2 && cross_2d(hull[hull.size() - 2], hull.back(), p) <= 0.0) {
            hull.pop_back();
        }
        hull.push_back(p);
    }
    const usize lower = hull.size() + 1;
    for (usize i = n; i-- > 0;) {
        const Point2D& p = pts[i];
        while (hull.size() >= lower && cross_2d(hull[hull.size() - 2], hull.back(), p) <= 0.0) {
            hull.pop_back();
        }
        hull.push_back(p);
    }
    hull.pop_back();
    const usize m = hull.size();
    if (m < 3) {
        return 0.0;
    }
    f64 s = 0.0;
    for (usize i = 0; i < m; ++i) {
        const Point2D& p = hull[i];
        const Point2D& q = hull[(i + 1) % m];
        s += p.x * q.y - q.x * p.y;
    }
    return 0.5 * abs(s);
}

struct CompAxis {
    f64 major = 0.0;
    f64 minor = 0.0;
    f64 eccentricity = 0.0;
    f64 orientation_deg = 0.0;
};

CompAxis compute_comp_axis(ArrayRef<i32> rs, ArrayRef<i32> cs) noexcept {
    const usize n = rs.size();
    CompAxis out{};
    if (n == 0) {
        return out;
    }
    f64 sr = 0.0;
    f64 sc = 0.0;
    for (usize i = 0; i < n; ++i) {
        sr += as<f64>(rs[i]);
        sc += as<f64>(cs[i]);
    }
    const f64 mr = sr / as<f64>(n);
    const f64 mc = sc / as<f64>(n);
    f64 m20 = 0.0;
    f64 m02 = 0.0;
    f64 m11 = 0.0;
    for (usize i = 0; i < n; ++i) {
        const f64 dc = as<f64>(cs[i]) - mc;
        const f64 dr = as<f64>(rs[i]) - mr;
        m20 += dc * dc;
        m02 += dr * dr;
        m11 += dc * dr;
    }
    const f64 a = m20 / as<f64>(n) + 1.0 / 12.0;
    const f64 b = m02 / as<f64>(n) + 1.0 / 12.0;
    const f64 cc = m11 / as<f64>(n);
    const f64 diff_sq = (a - b) * (a - b);
    const f64 sqrt_d = sqrt(diff_sq + 4.0 * cc * cc);
    const f64 lam1 = 0.5 * (a + b + sqrt_d);
    f64 lam2 = 0.5 * (a + b - sqrt_d);
    if (lam2 < 0.0) {
        lam2 = 0.0;
    }
    out.major = 4.0 * sqrt(lam1);
    out.minor = 4.0 * sqrt(lam2);
    out.eccentricity = (lam1 == 0.0) ? 0.0 : sqrt(1.0 - lam2 / lam1);
    f64 orient_rad = 0.0;
    if (cc == 0.0 && a == b) {
        orient_rad = 0.0;
    } else if (cc == 0.0) {
        orient_rad = (a < b) ? (kPi / 2.0) : 0.0;
    } else {
        orient_rad = -0.5 * atan2(2.0 * cc, a - b);
    }
    while (orient_rad > kPi / 2.0) {
        orient_rad -= kPi;
    }
    while (orient_rad <= -kPi / 2.0) {
        orient_rad += kPi;
    }
    out.orientation_deg = orient_rad * 180.0 / kPi;
    return out;
}

f64 comp_perimeter_th(ArrayRef<i32> rs, ArrayRef<i32> cs, i32 rmin, i32 rmax, i32 cmin, i32 cmax,
                      Allocator* a) {
    const i32 h = rmax - rmin + 1;
    const i32 w = cmax - cmin + 1;
    if (h <= 0 || w <= 0) {
        return 0.0;
    }
    const i32 hp = h + 2;
    const i32 wp = w + 2;
    Array<u8> mask(a);
    mask.assign(as<usize>(hp) * as<usize>(wp), u8{0});
    for (usize i = 0; i < rs.size(); ++i) {
        const i32 r = rs[i] - rmin + 1;
        const i32 c = cs[i] - cmin + 1;
        mask[as<usize>(r * wp + c)] = 1;
    }
    f64 sum = 0.0;
    for (i32 r = 0; r < hp - 1; ++r) {
        for (i32 c = 0; c < wp - 1; ++c) {
            const i32 p = (mask[as<usize>(r * wp + c)] ? 8 : 0)
                        | (mask[as<usize>(r * wp + c + 1)] ? 4 : 0)
                        | (mask[as<usize>((r + 1) * wp + c)] ? 2 : 0)
                        | (mask[as<usize>((r + 1) * wp + c + 1)] ? 1 : 0);
            sum += kPerimWeights[p];
        }
    }
    return sum;
}

f64 comp_perimeter_old(ArrayRef<i32> rs, ArrayRef<i32> cs, i32 rmin, i32 rmax, i32 cmin, i32 cmax,
                       Allocator* a) {
    const i32 h = rmax - rmin + 1;
    const i32 w = cmax - cmin + 1;
    if (h <= 0 || w <= 0) {
        return 0.0;
    }
    const i32 hp = h + 2;
    const i32 wp = w + 2;
    Array<u8> mask(a);
    mask.assign(as<usize>(hp) * as<usize>(wp), u8{0});
    for (usize i = 0; i < rs.size(); ++i) {
        const i32 r = rs[i] - rmin + 1;
        const i32 c = cs[i] - cmin + 1;
        mask[as<usize>(r * wp + c)] = 1;
    }
    i32 per = 0;
    for (i32 r = 1; r < hp - 1; ++r) {
        for (i32 c = 1; c < wp - 1; ++c) {
            if (!mask[as<usize>(r * wp + c)]) {
                continue;
            }
            if (!mask[as<usize>(r * wp + c - 1)]) {
                ++per;
            }
            if (!mask[as<usize>(r * wp + c + 1)]) {
                ++per;
            }
            if (!mask[as<usize>((r - 1) * wp + c)]) {
                ++per;
            }
            if (!mask[as<usize>((r + 1) * wp + c)]) {
                ++per;
            }
        }
    }
    return as<f64>(per);
}

f64 comp_euler(ArrayRef<i32> rs, ArrayRef<i32> cs, i32 rmin, i32 rmax, i32 cmin, i32 cmax, i32 conn,
               Allocator* a) {
    const i32 h = rmax - rmin + 1;
    const i32 w = cmax - cmin + 1;
    if (h <= 0 || w <= 0) {
        return 0.0;
    }
    const i32 hp = h + 2;
    const i32 wp = w + 2;
    Array<u8> mask(a);
    mask.assign(as<usize>(hp) * as<usize>(wp), u8{0});
    for (usize i = 0; i < rs.size(); ++i) {
        mask[as<usize>((rs[i] - rmin + 1) * wp + (cs[i] - cmin + 1))] = 1;
    }
    i32 n_q1 = 0;
    i32 n_q3 = 0;
    i32 n_qd = 0;
    for (i32 r = 0; r < hp - 1; ++r) {
        for (i32 c = 0; c < wp - 1; ++c) {
            const i32 p = (mask[as<usize>(r * wp + c)] ? 8 : 0)
                        | (mask[as<usize>(r * wp + c + 1)] ? 4 : 0)
                        | (mask[as<usize>((r + 1) * wp + c)] ? 2 : 0)
                        | (mask[as<usize>((r + 1) * wp + c + 1)] ? 1 : 0);
            const i32 pc = ((p >> 3) & 1) + ((p >> 2) & 1) + ((p >> 1) & 1) + (p & 1);
            if (pc == 1) {
                ++n_q1;
            } else if (pc == 3) {
                ++n_q3;
            } else if (p == 9 || p == 6) {
                ++n_qd;
            }
        }
    }
    if (conn == 4) {
        return as<f64>(n_q1 - n_q3 + 2 * n_qd) / 4.0;
    }
    return as<f64>(n_q1 - n_q3 - 2 * n_qd) / 4.0;
}

f64 comp_filled_area(ArrayRef<i32> rs, ArrayRef<i32> cs, i32 rmin, i32 rmax, i32 cmin, i32 cmax,
                     Allocator* a) {
    const i32 h = rmax - rmin + 1;
    const i32 w = cmax - cmin + 1;
    if (h <= 0 || w <= 0) {
        return 0.0;
    }
    const i32 hp = h + 2;
    const i32 wp = w + 2;
    const usize total = as<usize>(hp) * as<usize>(wp);
    Array<u8> mask(a);
    mask.assign(total, u8{0});
    for (usize i = 0; i < rs.size(); ++i) {
        mask[as<usize>((rs[i] - rmin + 1) * wp + (cs[i] - cmin + 1))] = 1;
    }
    Array<u8> visited(a);
    visited.assign(total, u8{0});
    Array<i32> queue(a);
    queue.reserve(total);
    auto push_if = [&](i32 r, i32 c) {
        if (r < 0 || c < 0 || r >= hp || c >= wp) {
            return;
        }
        const usize idx = as<usize>(r * wp + c);
        if (mask[idx] || visited[idx]) {
            return;
        }
        visited[idx] = 1;
        queue.push_back(as<i32>(idx));
    };
    for (i32 r = 0; r < hp; ++r) {
        push_if(r, 0);
        push_if(r, wp - 1);
    }
    for (i32 c = 0; c < wp; ++c) {
        push_if(0, c);
        push_if(hp - 1, c);
    }
    usize head = 0;
    while (head < queue.size()) {
        const i32 idx = queue[head++];
        const i32 r = idx / wp;
        const i32 c = idx - r * wp;
        push_if(r - 1, c);
        push_if(r + 1, c);
        push_if(r, c - 1);
        push_if(r, c + 1);
    }
    usize filled = 0;
    for (usize i = 0; i < total; ++i) {
        if (mask[i] || !visited[i]) {
            ++filled;
        }
    }
    return as<f64>(filled) - as<f64>(hp * wp - h * w);
}

f64 comp_convex_area(ArrayRef<i32> rs, ArrayRef<i32> cs, Allocator* a) {
    const usize n = rs.size();
    if (n == 0) {
        return 0.0;
    }
    Array<Point2D> pts(a);
    pts.reserve(4 * n);
    for (usize i = 0; i < n; ++i) {
        const f64 r = as<f64>(rs[i]);
        const f64 c = as<f64>(cs[i]);
        pts.push_back(Point2D{c - 0.5, r - 0.5});
        pts.push_back(Point2D{c + 0.5, r - 0.5});
        pts.push_back(Point2D{c - 0.5, r + 0.5});
        pts.push_back(Point2D{c + 0.5, r + 0.5});
    }
    return convex_hull_area(pts, a);
}

} // namespace

ImageStatus bwlabel_2d(ArrayRef<u8> bw_col_major, usize height, usize width, i32 conn,
                       MutArrayRef<i32> labels_col_major, i32& num_labels_out, Allocator* alloc) {
    const usize total = height * width;
    if (bw_col_major.size() != total || labels_col_major.size() != total) {
        return ImageStatus::kInvalidArgument;
    }
    if (conn != 4 && conn != 8) {
        conn = 8;
    }
    fill(labels_col_major.begin(), labels_col_major.end(), i32{0});
    num_labels_out = 0;
    if (total == 0) {
        return ImageStatus::kOk;
    }

    Allocator* a = alloc ? alloc : default_allocator();
    Array<i32> parent(a);
    parent.reserve(64);
    parent.push_back(0);

    auto find = [&](i32 x) noexcept -> i32 {
        while (parent[as<usize>(x)] != x) {
            parent[as<usize>(x)] = parent[as<usize>(parent[as<usize>(x)])];
            x = parent[as<usize>(x)];
        }
        return x;
    };
    auto unite = [&](i32 x, i32 y) noexcept {
        x = find(x);
        y = find(y);
        if (x != y) {
            parent[as<usize>(x)] = y;
        }
    };

    i32 next_label = 1;
    for (usize c = 0; c < width; ++c) {
        const usize col_off = c * height;
        for (usize r = 0; r < height; ++r) {
            const usize k = col_off + r;
            if (!bw_col_major[k]) {
                continue;
            }
            i32 nb[4] = {0, 0, 0, 0};
            i32 nc = 0;
            if (c > 0) {
                const i32 v = labels_col_major[(c - 1) * height + r];
                if (v) {
                    nb[nc++] = v;
                }
            }
            if (r > 0) {
                const i32 v = labels_col_major[col_off + (r - 1)];
                if (v) {
                    nb[nc++] = v;
                }
            }
            if (conn == 8) {
                if (c > 0 && r > 0) {
                    const i32 v = labels_col_major[(c - 1) * height + (r - 1)];
                    if (v) {
                        nb[nc++] = v;
                    }
                }
                if (c > 0 && r + 1 < height) {
                    const i32 v = labels_col_major[(c - 1) * height + (r + 1)];
                    if (v) {
                        nb[nc++] = v;
                    }
                }
            }
            if (nc == 0) {
                parent.push_back(next_label);
                labels_col_major[k] = next_label;
                ++next_label;
            } else {
                i32 m = nb[0];
                for (i32 i = 1; i < nc; ++i) {
                    if (nb[i] < m) {
                        m = nb[i];
                    }
                }
                labels_col_major[k] = m;
                for (i32 i = 0; i < nc; ++i) {
                    unite(nb[i], m);
                }
            }
        }
    }

    Array<i32> remap(a);
    remap.assign(parent.size(), i32{0});
    i32 num_k = 0;
    for (usize i = 0; i < total; ++i) {
        const i32 v = labels_col_major[i];
        if (v == 0) {
            continue;
        }
        const i32 root = find(v);
        if (remap[as<usize>(root)] == 0) {
            remap[as<usize>(root)] = ++num_k;
        }
        labels_col_major[i] = remap[as<usize>(root)];
    }
    num_labels_out = num_k;
    return ImageStatus::kOk;
}

ImageStatus bwlabel_3d(ArrayRef<u8> vol_col_major, usize height, usize width, usize depth, i32 conn,
                       MutArrayRef<i32> labels_col_major, i32& num_labels_out, Allocator* alloc) {
    const usize total = height * width * depth;
    if (vol_col_major.size() != total || labels_col_major.size() != total) {
        return ImageStatus::kInvalidArgument;
    }
    fill(labels_col_major.begin(), labels_col_major.end(), i32{0});
    num_labels_out = 0;
    if (total == 0) {
        return ImageStatus::kOk;
    }
    if (conn == 0) {
        conn = (depth > 1) ? 26 : 8;
    }
    const i32 max_dist = (depth > 1) ? ((conn == 6) ? 1 : (conn == 18 ? 2 : 3))
                                     : ((conn == 4) ? 1 : 2);

    Allocator* a = alloc ? alloc : default_allocator();
    Array<i32> parent(a);
    parent.push_back(0);
    Array<i32> sz(a);
    sz.push_back(0);

    auto find = [&](i32 x) noexcept -> i32 {
        while (parent[as<usize>(x)] != x) {
            parent[as<usize>(x)] = parent[as<usize>(parent[as<usize>(x)])];
            x = parent[as<usize>(x)];
        }
        return x;
    };

    const i32 h = as<i32>(height);
    const i32 w = as<i32>(width);
    const i32 d_max = as<i32>(depth);
    i32 next_label = 1;
    for (i32 d = 0; d < d_max; ++d) {
        for (i32 c = 0; c < w; ++c) {
            for (i32 r = 0; r < h; ++r) {
                const usize idx = (as<usize>(d) * width + as<usize>(c)) * height + as<usize>(r);
                if (!vol_col_major[idx]) {
                    continue;
                }
                const i32 node = next_label++;
                parent.push_back(node);
                sz.push_back(1);
                labels_col_major[idx] = node;
                for (i32 dd = -1; dd <= 0; ++dd) {
                    if (d + dd < 0) {
                        continue;
                    }
                    for (i32 dc = -1; dc <= (dd == 0 ? 0 : 1); ++dc) {
                        if (c + dc < 0 || c + dc >= w) {
                            continue;
                        }
                        for (i32 dr = -1; dr <= ((dd == 0 && dc == 0) ? -1 : 1); ++dr) {
                            if (r + dr < 0 || r + dr >= h) {
                                continue;
                            }
                            if (abs(dr) + abs(dc) + abs(dd) > max_dist) {
                                continue;
                            }
                            const usize nidx =
                                (as<usize>(d + dd) * width + as<usize>(c + dc)) * height
                                + as<usize>(r + dr);
                            const i32 nl = labels_col_major[nidx];
                            if (nl) {
                                const i32 x = find(node);
                                const i32 y = find(nl);
                                if (x != y) {
                                    if (sz[as<usize>(x)] > sz[as<usize>(y)]) {
                                        parent[as<usize>(y)] = x;
                                        sz[as<usize>(x)] += sz[as<usize>(y)];
                                    } else {
                                        parent[as<usize>(x)] = y;
                                        sz[as<usize>(y)] += sz[as<usize>(x)];
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    Array<i32> remap(a);
    remap.assign(parent.size(), i32{0});
    i32 num_k = 0;
    for (usize k = 1; k < parent.size(); ++k) {
        if (parent[k] == as<i32>(k)) {
            remap[k] = ++num_k;
        }
    }
    for (usize i = 0; i < total; ++i) {
        if (labels_col_major[i] == 0) {
            continue;
        }
        labels_col_major[i] = remap[as<usize>(find(labels_col_major[i]))];
    }
    num_labels_out = num_k;
    return ImageStatus::kOk;
}

ImageStatus bwperim_2d(ArrayRef<u8> bw_col_major, usize height, usize width, i32 conn,
                       MutArrayRef<u8> out_col_major) noexcept {
    const usize total = height * width;
    if (bw_col_major.size() != total || out_col_major.size() != total) {
        return ImageStatus::kInvalidArgument;
    }
    if (conn != 4) {
        conn = 8;
    }
    if (total == 0) {
        return ImageStatus::kOk;
    }
    const i32 h = as<i32>(height);
    const i32 w = as<i32>(width);
    for (i32 c = 0; c < w; ++c) {
        const usize col_off = as<usize>(c) * height;
        for (i32 r = 0; r < h; ++r) {
            const usize idx = col_off + as<usize>(r);
            if (!bw_col_major[idx]) {
                out_col_major[idx] = 0;
                continue;
            }
            bool boundary = false;
            for (i32 dr = -1; dr <= 1 && !boundary; ++dr) {
                for (i32 dc = -1; dc <= 1 && !boundary; ++dc) {
                    if (dr == 0 && dc == 0) {
                        continue;
                    }
                    if (conn == 4 && abs(dr) + abs(dc) > 1) {
                        continue;
                    }
                    const i32 rr = r + dr;
                    const i32 cc = c + dc;
                    if (rr < 0 || rr >= h || cc < 0 || cc >= w) {
                        boundary = true;
                        break;
                    }
                    if (!bw_col_major[as<usize>(cc) * height + as<usize>(rr)]) {
                        boundary = true;
                    }
                }
            }
            out_col_major[idx] = boundary ? u8{1} : u8{0};
        }
    }
    return ImageStatus::kOk;
}

ImageStatus bwareaopen_2d(ArrayRef<u8> bw_col_major, usize height, usize width, i32 min_area,
                          i32 conn, MutArrayRef<u8> out_col_major, Allocator* alloc) {
    const usize total = height * width;
    if (bw_col_major.size() != total || out_col_major.size() != total) {
        return ImageStatus::kInvalidArgument;
    }
    if (min_area < 1) {
        min_area = 1;
    }
    if (total == 0) {
        return ImageStatus::kOk;
    }
    Allocator* a = alloc ? alloc : default_allocator();
    Array<i32> labels(a);
    labels.resize(total);
    i32 num_k = 0;
    const ImageStatus st =
        bwlabel_2d(bw_col_major, height, width, conn, labels.as_mut_ref(), num_k, a);
    if (st != ImageStatus::kOk) {
        return st;
    }
    Array<i32> count(a);
    count.assign(as<usize>(num_k) + 1, i32{0});
    for (usize i = 0; i < total; ++i) {
        if (labels[i] > 0) {
            ++count[as<usize>(labels[i])];
        }
    }
    for (usize i = 0; i < total; ++i) {
        const i32 lab = labels[i];
        out_col_major[i] = (lab > 0 && count[as<usize>(lab)] >= min_area) ? u8{1} : u8{0};
    }
    return ImageStatus::kOk;
}

ImageStatus bwareafilt_2d(ArrayRef<u8> bw_col_major, usize height, usize width, f64 lo, f64 hi,
                          usize n_keep, bool keep_largest, i32 conn, MutArrayRef<u8> out_col_major,
                          Allocator* alloc) {
    const usize total = height * width;
    if (bw_col_major.size() != total || out_col_major.size() != total) {
        return ImageStatus::kInvalidArgument;
    }
    fill(out_col_major.begin(), out_col_major.end(), u8{0});
    if (total == 0) {
        return ImageStatus::kOk;
    }
    Allocator* a = alloc ? alloc : default_allocator();
    Array<i32> labels(a);
    labels.resize(total);
    i32 num_k = 0;
    const ImageStatus st =
        bwlabel_2d(bw_col_major, height, width, conn, labels.as_mut_ref(), num_k, a);
    if (st != ImageStatus::kOk || num_k == 0) {
        return st;
    }

    Array<usize> area(a);
    area.assign(as<usize>(num_k) + 1, usize{0});
    for (usize i = 0; i < total; ++i) {
        if (labels[i] > 0) {
            ++area[as<usize>(labels[i])];
        }
    }

    Array<u8> keep(a);
    keep.assign(as<usize>(num_k) + 1, u8{0});
    if (n_keep > 0) {
        Array<i32> idx(a);
        idx.reserve(as<usize>(num_k));
        for (i32 k = 1; k <= num_k; ++k) {
            idx.push_back(k);
        }
        if (keep_largest) {
            sort(idx.begin(), idx.end(), [&](i32 x, i32 y) noexcept {
                if (area[as<usize>(x)] != area[as<usize>(y)]) {
                    return area[as<usize>(x)] > area[as<usize>(y)];
                }
                return x < y;
            });
        } else {
            sort(idx.begin(), idx.end(), [&](i32 x, i32 y) noexcept {
                if (area[as<usize>(x)] != area[as<usize>(y)]) {
                    return area[as<usize>(x)] < area[as<usize>(y)];
                }
                return x < y;
            });
        }
        const usize take = min(n_keep, idx.size());
        for (usize i = 0; i < take; ++i) {
            keep[as<usize>(idx[i])] = 1;
        }
    } else {
        for (i32 k = 1; k <= num_k; ++k) {
            const f64 ar = as<f64>(area[as<usize>(k)]);
            keep[as<usize>(k)] = (ar >= lo && ar <= hi) ? u8{1} : u8{0};
        }
    }

    for (usize i = 0; i < total; ++i) {
        const i32 k = labels[i];
        out_col_major[i] = (k > 0 && keep[as<usize>(k)]) ? u8{1} : u8{0};
    }
    return ImageStatus::kOk;
}

ImageStatus bwselect_2d(ArrayRef<u8> bw_col_major, usize height, usize width,
                        ArrayRef<i32> seed_rows, ArrayRef<i32> seed_cols, i32 conn,
                        MutArrayRef<u8> out_col_major, Allocator* alloc) {
    const usize total = height * width;
    if (bw_col_major.size() != total || out_col_major.size() != total) {
        return ImageStatus::kInvalidArgument;
    }
    fill(out_col_major.begin(), out_col_major.end(), u8{0});
    if (total == 0) {
        return ImageStatus::kOk;
    }
    Allocator* a = alloc ? alloc : default_allocator();
    Array<i32> labels(a);
    labels.resize(total);
    i32 num_k = 0;
    const ImageStatus st =
        bwlabel_2d(bw_col_major, height, width, conn, labels.as_mut_ref(), num_k, a);
    if (st != ImageStatus::kOk || num_k == 0) {
        return st;
    }

    const i32 h = as<i32>(height);
    const i32 w = as<i32>(width);
    const usize n_seed = min(seed_rows.size(), seed_cols.size());
    Array<u8> keep(a);
    keep.assign(as<usize>(num_k) + 1, u8{0});
    for (usize s = 0; s < n_seed; ++s) {
        const i32 r = seed_rows[s];
        const i32 c = seed_cols[s];
        if (r < 0 || r >= h || c < 0 || c >= w) {
            continue;
        }
        const i32 k = labels[as<usize>(c) * height + as<usize>(r)];
        if (k > 0) {
            keep[as<usize>(k)] = 1;
        }
    }
    for (usize i = 0; i < total; ++i) {
        const i32 k = labels[i];
        out_col_major[i] = (k > 0 && keep[as<usize>(k)]) ? u8{1} : u8{0};
    }
    return ImageStatus::kOk;
}

ImageStatus bweuler_2d(ArrayRef<u8> bw_col_major, usize height, usize width, i32 conn,
                       f64& euler_out) noexcept {
    if (bw_col_major.size() != height * width || (conn != 4 && conn != 8)) {
        return ImageStatus::kInvalidArgument;
    }
    static constexpr i32 kLut8[16] = {0, 1, 1, 0, 1, 0, -2, -1, 1, -2, 0, -1, 0, -1, -1, 0};
    static constexpr i32 kLut4[16] = {0, 1, 1, 0, 1, 0, 2, -1, 1, 2, 0, -1, 0, -1, -1, 0};
    const i32* lut = (conn == 4) ? kLut4 : kLut8;
    const i32 m = as<i32>(height);
    const i32 n = as<i32>(width);
    auto get = [&](i32 r, i32 c) noexcept -> i32 {
        if (r < 0 || c < 0 || r >= m || c >= n) {
            return 0;
        }
        return bw_col_major[as<usize>(c) * height + as<usize>(r)] ? 1 : 0;
    };
    i32 sum = 0;
    for (i32 r = 0; r <= m; ++r) {
        for (i32 c = 0; c <= n; ++c) {
            const i32 idx = get(r - 1, c - 1) + 2 * get(r, c - 1) + 4 * get(r - 1, c)
                          + 8 * get(r, c);
            sum += lut[idx];
        }
    }
    euler_out = as<f64>(sum) / 4.0;
    return ImageStatus::kOk;
}

ImageStatus bwdist_euclidean_2d(ArrayRef<u8> bw_col_major, usize height, usize width,
                                MutArrayRef<f64> out_dist_col_major, Allocator* alloc) {
    const usize total = height * width;
    if (bw_col_major.size() != total || out_dist_col_major.size() != total) {
        return ImageStatus::kInvalidArgument;
    }
    if (total == 0) {
        return ImageStatus::kOk;
    }
    Allocator* a = alloc ? alloc : default_allocator();
    const f64 inf = Limits<f64>::infinity();
    const usize max_dim = max(height, width);
    Array<f64> g(a);
    g.resize(total);
    Array<f64> line_in(a);
    line_in.resize(max_dim);
    Array<f64> line_out(a);
    line_out.resize(max_dim);
    Array<i64> vbuf(a);
    vbuf.resize(max_dim);
    Array<f64> zbuf(a);
    zbuf.resize(max_dim + 1);

    // Pass 1: horizontal DT along each row.
    for (usize r = 0; r < height; ++r) {
        for (usize c = 0; c < width; ++c) {
            line_in[c] = bw_col_major[c * height + r] ? 0.0 : inf;
        }
        dt_1d_parabolic(ArrayRef<f64>(line_in.data(), width),
                        MutArrayRef<f64>(line_out.data(), width),
                        MutArrayRef<i64>(vbuf.data(), width),
                        MutArrayRef<f64>(zbuf.data(), width + 1));
        for (usize c = 0; c < width; ++c) {
            g[c * height + r] = line_out[c];
        }
    }

    // Pass 2: vertical DT along each column (contiguous in col-major!).
    for (usize c = 0; c < width; ++c) {
        ArrayRef<f64> col_in(g.data() + c * height, height);
        MutArrayRef<f64> col_out(out_dist_col_major.data() + c * height, height);
        dt_1d_parabolic(col_in, col_out, MutArrayRef<i64>(vbuf.data(), height),
                        MutArrayRef<f64>(zbuf.data(), height + 1));
        for (usize r = 0; r < height; ++r) {
            const f64 sq = col_out[r];
            col_out[r] = is_inf(sq) ? inf : sqrt(sq);
        }
    }
    return ImageStatus::kOk;
}

ImageStatus bwboundaries_2d(ArrayRef<u8> bw_col_major, usize height, usize width, i32 conn,
                            MutArrayRef<i32> labels_col_major, i32& num_labels_out,
                            Array<BoundaryPoint>& points_out, Array<usize>& offsets_out,
                            Allocator* alloc) {
    const usize total = height * width;
    if (bw_col_major.size() != total || labels_col_major.size() != total) {
        return ImageStatus::kInvalidArgument;
    }
    if (conn != 4) {
        conn = 8;
    }
    points_out.clear();
    offsets_out.clear();
    Allocator* a = alloc ? alloc : default_allocator();
    const ImageStatus st =
        bwlabel_2d(bw_col_major, height, width, conn, labels_col_major, num_labels_out, a);
    if (st != ImageStatus::kOk) {
        return st;
    }
    offsets_out.push_back(0);
    if (num_labels_out == 0) {
        return ImageStatus::kOk;
    }

    const i32 h = as<i32>(height);
    const i32 w = as<i32>(width);
    // Start-pixel scan: first encounter in row-major order is topmost-leftmost.
    Array<i32> start_r(a);
    start_r.assign(as<usize>(num_labels_out) + 1, i32{-1});
    Array<i32> start_c(a);
    start_c.assign(as<usize>(num_labels_out) + 1, i32{-1});
    i32 remaining = num_labels_out;
    for (i32 r = 0; r < h && remaining > 0; ++r) {
        for (i32 c = 0; c < w; ++c) {
            const i32 lab = labels_col_major[as<usize>(c) * height + as<usize>(r)];
            if (lab > 0 && start_r[as<usize>(lab)] < 0) {
                start_r[as<usize>(lab)] = r;
                start_c[as<usize>(lab)] = c;
                --remaining;
            }
        }
    }

    static constexpr i32 kDr[8] = {0, 1, 1, 1, 0, -1, -1, -1};
    static constexpr i32 kDc[8] = {1, 1, 0, -1, -1, -1, 0, 1};
    auto is_fg = [&](i32 r, i32 c, i32 target) noexcept -> bool {
        if (r < 0 || r >= h || c < 0 || c >= w) {
            return false;
        }
        return labels_col_major[as<usize>(c) * height + as<usize>(r)] == target;
    };

    const usize cap = total * 4;
    for (i32 lab = 1; lab <= num_labels_out; ++lab) {
        const i32 r0 = start_r[as<usize>(lab)];
        const i32 c0 = start_c[as<usize>(lab)];
        points_out.push_back(BoundaryPoint{as<i64>(r0), as<i64>(c0)});

        bool has_nb = false;
        for (i32 d = 0; d < 8; ++d) {
            if (is_fg(r0 + kDr[d], c0 + kDc[d], lab)) {
                has_nb = true;
                break;
            }
        }
        if (!has_nb) {
            offsets_out.push_back(points_out.size());
            continue;
        }

        i32 prev_dir = (conn == 8) ? 7 : 6;
        i32 cur_r = r0;
        i32 cur_c = c0;
        usize iter = 0;
        while (iter++ < cap) {
            i32 found = -1;
            for (i32 step = 1; step <= 8; ++step) {
                const i32 d = (prev_dir + step) % 8;
                if (conn == 4 && (d & 1)) {
                    continue;
                }
                if (is_fg(cur_r + kDr[d], cur_c + kDc[d], lab)) {
                    found = d;
                    break;
                }
            }
            if (found < 0) {
                break;
            }
            const i32 nr = cur_r + kDr[found];
            const i32 nc = cur_c + kDc[found];
            prev_dir = (found + 4) % 8;
            points_out.push_back(BoundaryPoint{as<i64>(nr), as<i64>(nc)});
            if (nr == r0 && nc == c0) {
                break;
            }
            cur_r = nr;
            cur_c = nc;
        }
        offsets_out.push_back(points_out.size());
    }
    return ImageStatus::kOk;
}

f64 region_perimeter_2d(ArrayRef<i32> labels_col_major, usize height, usize width, i32 label,
                        i32 start_row, i32 start_col, Allocator* alloc) {
    const i32 h = as<i32>(height);
    const i32 w = as<i32>(width);
    if (labels_col_major.size() != height * width || start_row < 0 || start_row >= h
        || start_col < 0 || start_col >= w) {
        return 0.0;
    }
    static constexpr i32 kDr8[8] = {-1, -1, 0, 1, 1, 1, 0, -1};
    static constexpr i32 kDc8[8] = {0, 1, 1, 1, 0, -1, -1, -1};
    auto fg = [&](i32 r, i32 c) noexcept -> bool {
        return r >= 0 && r < h && c >= 0 && c < w
            && labels_col_major[as<usize>(c) * height + as<usize>(r)] == label;
    };
    Allocator* a = alloc ? alloc : default_allocator();
    Array<i32> br(a);
    Array<i32> bc(a);
    br.reserve(64);
    bc.reserve(64);
    br.push_back(start_row);
    bc.push_back(start_col);
    i32 back_dir = 6;
    i32 cr = start_row;
    i32 cc = start_col;
    const usize cap = height * width * 4 + 8;
    while (br.size() < cap) {
        bool found = false;
        for (i32 k = 0; k < 8; ++k) {
            const i32 d = (back_dir + k + 1) % 8;
            const i32 nr = cr + kDr8[d];
            const i32 nc = cc + kDc8[d];
            if (fg(nr, nc)) {
                br.push_back(nr);
                bc.push_back(nc);
                cr = nr;
                cc = nc;
                back_dir = (d + 4) % 8;
                found = true;
                break;
            }
        }
        if (!found) {
            break;
        }
        if (cr == start_row && cc == start_col && br.size() >= 2) {
            break;
        }
    }
    const usize n = br.size();
    if (n < 3) {
        return 0.0;
    }
    const usize steps = n - 1;
    auto step_type = [&](usize i) noexcept -> i32 {
        const i32 dr = br[i + 1] - br[i];
        const i32 dc = bc[i + 1] - bc[i];
        return (dr == 0) ? 0 : ((dc == 0) ? 1 : 2);
    };
    i64 ne = 0;
    i64 no = 0;
    i64 nc = 0;
    for (usize i = 0; i < steps; ++i) {
        const i32 t = step_type(i);
        if (t == 2) {
            ++no;
        } else {
            ++ne;
        }
        if (t != step_type((i + 1) % steps)) {
            ++nc;
        }
    }
    return 0.980 * as<f64>(ne) + 1.406 * as<f64>(no) - 0.091 * as<f64>(nc);
}

ImageStatus region_props_2d(ArrayRef<i32> labels_col_major, usize height, usize width,
                            i32 num_labels, u32 prop_flags, Array<RegionProps2D>& out_props,
                            Allocator* alloc) {
    if (labels_col_major.size() != height * width || num_labels < 0) {
        return ImageStatus::kInvalidArgument;
    }
    out_props.clear();
    if (num_labels == 0) {
        return ImageStatus::kOk;
    }
    Allocator* a = alloc ? alloc : default_allocator();
    const usize k_cnt = as<usize>(num_labels);
    out_props.resize(k_cnt);

    const bool need_moments = (prop_flags & kRegionPropEllipse) != 0;
    const bool need_perim = (prop_flags & kRegionPropPerimeter) != 0;

    Array<f64> sum_x(a);
    sum_x.assign(k_cnt, 0.0);
    Array<f64> sum_y(a);
    sum_y.assign(k_cnt, 0.0);
    Array<f64> sum_xx(a);
    Array<f64> sum_yy(a);
    Array<f64> sum_xy(a);
    if (need_moments) {
        sum_xx.assign(k_cnt, 0.0);
        sum_yy.assign(k_cnt, 0.0);
        sum_xy.assign(k_cnt, 0.0);
    }
    Array<i32> first_r(a);
    first_r.assign(k_cnt, i32{-1});
    Array<i32> first_c(a);
    first_c.assign(k_cnt, i32{-1});

    for (usize i = 0; i < k_cnt; ++i) {
        out_props[i].min_col = Limits<i32>::kMax;
        out_props[i].min_row = Limits<i32>::kMax;
        out_props[i].max_col = Limits<i32>::kMin;
        out_props[i].max_row = Limits<i32>::kMin;
    }

    const i32 h = as<i32>(height);
    const i32 w = as<i32>(width);
    // Scan row-major so first_r/first_c is topmost-leftmost per label.
    for (i32 r = 0; r < h; ++r) {
        for (i32 c = 0; c < w; ++c) {
            const i32 lab = labels_col_major[as<usize>(c) * height + as<usize>(r)];
            if (lab <= 0 || lab > num_labels) {
                continue;
            }
            const usize idx = as<usize>(lab - 1);
            RegionProps2D& p = out_props[idx];
            ++p.area;
            if (first_r[idx] < 0) {
                first_r[idx] = r;
                first_c[idx] = c;
            }
            const f64 cf = as<f64>(c);
            const f64 rf = as<f64>(r);
            sum_x[idx] += cf;
            sum_y[idx] += rf;
            if (need_moments) {
                sum_xx[idx] += cf * cf;
                sum_yy[idx] += rf * rf;
                sum_xy[idx] += cf * rf;
            }
            if (c < p.min_col) {
                p.min_col = c;
            }
            if (r < p.min_row) {
                p.min_row = r;
            }
            if (c > p.max_col) {
                p.max_col = c;
            }
            if (r > p.max_row) {
                p.max_row = r;
            }
        }
    }

    for (usize idx = 0; idx < k_cnt; ++idx) {
        RegionProps2D& p = out_props[idx];
        const f64 n = as<f64>(p.area);
        if (n > 0.0) {
            p.centroid_x = sum_x[idx] / n;
            p.centroid_y = sum_y[idx] / n;
            p.equiv_diameter = sqrt(4.0 * n / kPi);
            const f64 bw = as<f64>(p.max_col - p.min_col + 1);
            const f64 bh = as<f64>(p.max_row - p.min_row + 1);
            const f64 bba = bw * bh;
            p.extent = (bba > 0.0) ? (n / bba) : 0.0;
        }
        if (need_perim && p.area > 0 && first_r[idx] >= 0) {
            p.perimeter = region_perimeter_2d(labels_col_major, height, width, as<i32>(idx + 1),
                                              first_r[idx], first_c[idx], a);
        }
        if (need_moments && n > 0.0) {
            const f64 xbar = p.centroid_x;
            const f64 ybar = p.centroid_y;
            const f64 uxx = sum_xx[idx] / n - xbar * xbar + 1.0 / 12.0;
            const f64 uyy = sum_yy[idx] / n - ybar * ybar + 1.0 / 12.0;
            const f64 uxy = sum_xy[idx] / n - xbar * ybar;
            const f64 common = sqrt((uxx - uyy) * (uxx - uyy) + 4.0 * uxy * uxy);
            p.major_axis_length = 2.0 * sqrt(2.0) * sqrt(uxx + uyy + common);
            const f64 minor_arg = max(0.0, uxx + uyy - common);
            p.minor_axis_length = 2.0 * sqrt(2.0) * sqrt(minor_arg);
            if (p.major_axis_length > 0.0) {
                const f64 aa = p.major_axis_length / 2.0;
                const f64 bb = p.minor_axis_length / 2.0;
                const f64 d = aa * aa - bb * bb;
                p.eccentricity = 2.0 * sqrt(d > 0.0 ? d : 0.0) / p.major_axis_length;
            }
            if (uxy == 0.0) {
                p.orientation_deg = (uyy > uxx) ? 90.0 : 0.0;
            } else {
                const f64 uxy_m = -uxy;
                f64 num = 0.0;
                f64 den = 0.0;
                if (uyy > uxx) {
                    num = uyy - uxx + sqrt((uyy - uxx) * (uyy - uxx) + 4.0 * uxy_m * uxy_m);
                    den = 2.0 * uxy_m;
                } else {
                    num = 2.0 * uxy_m;
                    den = uxx - uyy + sqrt((uxx - uyy) * (uxx - uyy) + 4.0 * uxy_m * uxy_m);
                }
                p.orientation_deg = (180.0 / kPi) * atan(num / den);
            }
        }
    }
    return ImageStatus::kOk;
}

ImageStatus region_component_attrs(ArrayRef<i32> pixel_rows, ArrayRef<i32> pixel_cols,
                                   ArrayRef<usize> comp_offsets, usize height, usize width,
                                   i32 conn, RegionAttr attr, ArrayRef<f64> marker_col_major,
                                   MutArrayRef<f64> out_values, Allocator* alloc) {
    if (pixel_rows.size() != pixel_cols.size() || comp_offsets.size() == 0) {
        return ImageStatus::kInvalidArgument;
    }
    const usize num_comps = comp_offsets.size() - 1;
    if (out_values.size() != num_comps) {
        return ImageStatus::kInvalidArgument;
    }
    const bool needs_marker = (attr == RegionAttr::kMinIntensity
                               || attr == RegionAttr::kMaxIntensity
                               || attr == RegionAttr::kMeanIntensity);
    if (needs_marker && marker_col_major.size() != height * width) {
        return ImageStatus::kInvalidArgument;
    }
    Allocator* a = alloc ? alloc : default_allocator();
    for (usize k = 0; k < num_comps; ++k) {
        const usize beg = comp_offsets[k];
        const usize end = comp_offsets[k + 1];
        if (beg > end || end > pixel_rows.size()) {
            return ImageStatus::kInvalidArgument;
        }
        const usize n = end - beg;
        if (n == 0) {
            out_values[k] = (attr == RegionAttr::kCircularity || attr == RegionAttr::kSolidity ||
                             attr == RegionAttr::kMinIntensity || attr == RegionAttr::kMaxIntensity ||
                             attr == RegionAttr::kMeanIntensity)
                                ? kNaN
                                : 0.0;
            continue;
        }
        ArrayRef<i32> rs(pixel_rows.data() + beg, n);
        ArrayRef<i32> cs(pixel_cols.data() + beg, n);
        i32 rmin = Limits<i32>::kMax;
        i32 rmax = Limits<i32>::kMin;
        i32 cmin = Limits<i32>::kMax;
        i32 cmax = Limits<i32>::kMin;
        for (usize i = 0; i < n; ++i) {
            if (needs_marker &&
                (rs[i] < 0 || cs[i] < 0 || as<usize>(rs[i]) >= height ||
                 as<usize>(cs[i]) >= width)) {
                return ImageStatus::kInvalidArgument;
            }
            if (rs[i] < rmin) {
                rmin = rs[i];
            }
            if (rs[i] > rmax) {
                rmax = rs[i];
            }
            if (cs[i] < cmin) {
                cmin = cs[i];
            }
            if (cs[i] > cmax) {
                cmax = cs[i];
            }
        }
        f64 val = 0.0;
        switch (attr) {
            case RegionAttr::kArea:
                val = as<f64>(n);
                break;
            case RegionAttr::kEquivDiameter:
                val = 2.0 * sqrt(as<f64>(n) / kPi);
                break;
            case RegionAttr::kExtent: {
                const i32 bw = cmax - cmin + 1;
                const i32 bh = rmax - rmin + 1;
                val = (bw > 0 && bh > 0) ? (as<f64>(n) / (as<f64>(bw) * as<f64>(bh))) : 0.0;
                break;
            }
            case RegionAttr::kMajorAxisLength:
                val = compute_comp_axis(rs, cs).major;
                break;
            case RegionAttr::kMinorAxisLength:
                val = compute_comp_axis(rs, cs).minor;
                break;
            case RegionAttr::kEccentricity:
                val = compute_comp_axis(rs, cs).eccentricity;
                break;
            case RegionAttr::kOrientation:
                val = compute_comp_axis(rs, cs).orientation_deg;
                break;
            case RegionAttr::kPerimeter:
                val = comp_perimeter_th(rs, cs, rmin, rmax, cmin, cmax, a);
                break;
            case RegionAttr::kPerimeterOld:
                val = comp_perimeter_old(rs, cs, rmin, rmax, cmin, cmax, a);
                break;
            case RegionAttr::kCircularity: {
                const f64 per = comp_perimeter_th(rs, cs, rmin, rmax, cmin, cmax, a);
                val = (per > 0.0) ? (4.0 * kPi * as<f64>(n) / (per * per)) : kNaN;
                break;
            }
            case RegionAttr::kConvexArea:
                val = comp_convex_area(rs, cs, a);
                break;
            case RegionAttr::kSolidity: {
                const f64 ca = comp_convex_area(rs, cs, a);
                val = (ca > 0.0) ? (as<f64>(n) / ca) : kNaN;
                break;
            }
            case RegionAttr::kEulerNumber:
                val = comp_euler(rs, cs, rmin, rmax, cmin, cmax, conn, a);
                break;
            case RegionAttr::kFilledArea:
                val = comp_filled_area(rs, cs, rmin, rmax, cmin, cmax, a);
                break;
            case RegionAttr::kMinIntensity: {
                if (n == 0) {
                    val = kNaN;
                } else {
                    f64 m = Limits<f64>::infinity();
                    for (usize i = 0; i < n; ++i) {
                        const f64 v = marker_col_major[as<usize>(cs[i]) * height + as<usize>(rs[i])];
                        if (v < m) {
                            m = v;
                        }
                    }
                    val = m;
                }
                break;
            }
            case RegionAttr::kMaxIntensity: {
                if (n == 0) {
                    val = kNaN;
                } else {
                    f64 m = -Limits<f64>::infinity();
                    for (usize i = 0; i < n; ++i) {
                        const f64 v = marker_col_major[as<usize>(cs[i]) * height + as<usize>(rs[i])];
                        if (v > m) {
                            m = v;
                        }
                    }
                    val = m;
                }
                break;
            }
            case RegionAttr::kMeanIntensity: {
                if (n == 0) {
                    val = kNaN;
                } else {
                    f64 s = 0.0;
                    for (usize i = 0; i < n; ++i) {
                        s += marker_col_major[as<usize>(cs[i]) * height + as<usize>(rs[i])];
                    }
                    val = s / as<f64>(n);
                }
                break;
            }
        }
        out_values[k] = val;
    }
    return ImageStatus::kOk;
}

ImageStatus freeman_chain_code(ArrayRef<f64> rows, ArrayRef<f64> cols, Array<f64>& out_fcc,
                               Array<f64>& out_diff) {
    if (rows.size() != cols.size()) {
        return ImageStatus::kInvalidArgument;
    }
    out_fcc.clear();
    out_diff.clear();
    const usize n = rows.size();
    if (n == 0) {
        return ImageStatus::kOk;
    }
    const bool closed = (rows[0] == rows[n - 1] && cols[0] == cols[n - 1]);
    const usize n_codes = closed ? (n - 1) : n;
    out_fcc.resize(n_codes);
    out_diff.resize(n_codes);

    auto dir_code = [](i32 dr, i32 dc) noexcept -> f64 {
        if (dr == -1 && dc == -1) return 3.0;
        if (dr == -1 && dc == 0) return 2.0;
        if (dr == -1 && dc == 1) return 1.0;
        if (dr == 0 && dc == -1) return 4.0;
        if (dr == 0 && dc == 1) return 0.0;
        if (dr == 1 && dc == -1) return 5.0;
        if (dr == 1 && dc == 0) return 6.0;
        if (dr == 1 && dc == 1) return 7.0;
        return kNaN;
    };

    for (usize k = 0; k < n_codes; ++k) {
        const usize next = (k + 1 < n) ? (k + 1) : 0;
        const f64 df_r = rows[next] - rows[k];
        const f64 df_c = cols[next] - cols[k];
        if (!is_finite(df_r) || !is_finite(df_c) || abs(df_r) > 1.5 || abs(df_c) > 1.5) {
            out_fcc[k] = kNaN;
            continue;
        }
        const i32 dr = as<i32>(round(df_r));
        const i32 dc = as<i32>(round(df_c));
        out_fcc[k] = dir_code(dr, dc);
    }
    for (usize k = 0; k < n_codes; ++k) {
        const f64 a = out_fcc[k];
        const f64 b = out_fcc[(k + 1) % n_codes];
        f64 m = fmod(b - a, 8.0);
        if (m < 0.0) {
            m += 8.0;
        }
        out_diff[k] = m;
    }
    return ImageStatus::kOk;
}

} // namespace keel::image
