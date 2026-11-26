// keelcpp/image/src/morph.cpp
// Pure zero-STL 2D/3D morphological operations.

#include "keel/image.h"

#include "bwmorph_luts.h"
#include "keel/arena.h"
#include "keel/array.h"
#include "keel/hash_set.h"
#include "keel/math.h"
#include "keel/static_array.h"

namespace keel::image {

namespace {

constexpr f64 kPi = 3.14159265358979323846;

struct GridPt {
    i32 r = 0;
    i32 c = 0;
    bool operator==(const GridPt& o) const noexcept { return r == o.r && c == o.c; }
};

struct GridPtHash {
    u64 operator()(const GridPt& p) const noexcept {
        const u64 k = (as<u64>(as<u32>(p.r)) << 32) | as<u64>(as<u32>(p.c));
        return Hash<u64>{}(k);
    }
};

using GridPtSet = HashSet<GridPt, GridPtHash>;

void minksum_periodic(GridPtSet& pts, i32 vr, i32 vc, i32 rp, Allocator* alloc) {
    GridPtSet out(alloc);
    for (const GridPt& p : pts) {
        for (i32 j = -rp; j <= rp; ++j) {
            out.insert(GridPt{p.r + j * vr, p.c + j * vc});
        }
    }
    pts = move(out);
}

ImageStatus strel_disk_euclidean(f64 r, StrelMask& out, Allocator* alloc) {
    const i32 rad = as<i32>(ceil(r));
    const i32 n = 2 * rad + 1;
    const usize nu = as<usize>(n);
    out.rows = nu;
    out.cols = nu;
    out.mask_col_major = Array<u8>(nu * nu, u8{0}, alloc);
    const f64 r2 = r * r;
    for (i32 c = 0; c < n; ++c) {
        const f64 dx = as<f64>(c - rad);
        for (i32 r_idx = 0; r_idx < n; ++r_idx) {
            const f64 dy = as<f64>(r_idx - rad);
            if (dy * dy + dx * dx <= r2) {
                out.mask_col_major[as<usize>(c) * nu + as<usize>(r_idx)] = 1;
            }
        }
    }
    return ImageStatus::kOk;
}

template <bool IsErode>
void block_van_herk(const u8* in, u8* out, i32 n, i32 lo, i32 hi, u8* padded, u8* prefix,
                    u8* suffix) noexcept {
    auto reduce = [](u8 a, u8 b) noexcept -> u8 { return IsErode ? min(a, b) : max(a, b); };
    const i32 k = hi - lo + 1;
    const u8 pad = IsErode ? u8{255} : u8{0};
    const i32 off = -lo;
    const i32 m = n + k - 1;

    for (i32 i = 0; i < off; ++i) {
        padded[i] = pad;
    }
    for (i32 i = 0; i < n; ++i) {
        padded[off + i] = in[i];
    }
    for (i32 i = off + n; i < m; ++i) {
        padded[i] = pad;
    }

    for (i32 b = 0; b < m; b += k) {
        const i32 e = min(b + k, m);
        prefix[b] = padded[b];
        for (i32 i = b + 1; i < e; ++i) {
            prefix[i] = reduce(prefix[i - 1], padded[i]);
        }
    }
    for (i32 b = 0; b < m; b += k) {
        const i32 e = min(b + k, m);
        suffix[e - 1] = padded[e - 1];
        for (i32 i = e - 2; i >= b; --i) {
            suffix[i] = reduce(suffix[i + 1], padded[i]);
        }
    }
    for (i32 r = 0; r < n; ++r) {
        out[r] = reduce(suffix[r], prefix[r + k - 1]);
    }
}

bool van_herk_eligible(ArrayRef<u8> se_col_major, i32 se_h, i32 se_w, i32 half_r,
                       i32 half_c) noexcept {
    if (half_r < 0 || half_c < 0 || half_r >= se_h || half_c >= se_w) {
        return false;
    }
    if (se_col_major[as<usize>(half_c) * as<usize>(se_h) + as<usize>(half_r)] == 0) {
        return false;
    }
    for (i32 kj = 0; kj < se_w; ++kj) {
        i32 lo = -1;
        i32 hi = -1;
        for (i32 ki = 0; ki < se_h; ++ki) {
            if (se_col_major[as<usize>(kj) * as<usize>(se_h) + as<usize>(ki)] != 0) {
                if (lo < 0) {
                    lo = ki;
                }
                hi = ki;
            }
        }
        for (i32 ki = lo; ki >= 0 && ki <= hi; ++ki) {
            if (se_col_major[as<usize>(kj) * as<usize>(se_h) + as<usize>(ki)] == 0) {
                return false;
            }
        }
    }
    return true;
}

template <bool IsErode>
ImageStatus morph_flat_u8_impl(ArrayRef<u8> src_col_major, usize height, usize width,
                               ArrayRef<u8> se_col_major, usize se_h, usize se_w,
                               MutArrayRef<u8> dst_col_major, Allocator* alloc) {
    const usize n = height * width;
    if (src_col_major.size() < n || dst_col_major.size() < n ||
        se_col_major.size() < se_h * se_w) {
        return ImageStatus::kInvalidArgument;
    }
    if (n == 0) {
        return ImageStatus::kOk;
    }

    const i32 h = as<i32>(height);
    const i32 w = as<i32>(width);
    const i32 sh = as<i32>(se_h);
    const i32 sw = as<i32>(se_w);
    const i32 half_r = sh / 2;
    const i32 half_c = sw / 2;
    const u8* src = src_col_major.data();
    u8* dst = dst_col_major.data();

    ArenaTemp scratch = get_scratch(alloc);
    Array<u8> src_copy(scratch.allocator());
    if (src == dst) {
        src_copy.resize(n);
        for (usize i = 0; i < n; ++i) {
            src_copy[i] = src[i];
        }
        src = src_copy.data();
    }

    if (van_herk_eligible(se_col_major, sh, sw, half_r, half_c)) {
        const u8 init_val = IsErode ? u8{255} : u8{0};
        for (usize i = 0; i < n; ++i) {
            dst[i] = init_val;
        }
        struct Run {
            i32 lo;
            i32 hi;
        };
        Array<Run> runs(scratch.allocator());
        Array<i32> col_run(scratch.allocator());
        Array<i32> col_dc(scratch.allocator());
        i32 max_k = 1;
        for (i32 kj = 0; kj < sw; ++kj) {
            i32 ki_lo = -1;
            i32 ki_hi = -1;
            for (i32 ki = 0; ki < sh; ++ki) {
                if (se_col_major[as<usize>(kj) * as<usize>(sh) + as<usize>(ki)] != 0) {
                    if (ki_lo < 0) {
                        ki_lo = ki;
                    }
                    ki_hi = ki;
                }
            }
            if (ki_lo < 0) {
                continue;
            }
            const i32 lo = ki_lo - half_r;
            const i32 hi = ki_hi - half_r;
            i32 rid = -1;
            for (i32 t = 0; t < as<i32>(runs.size()); ++t) {
                if (runs[as<usize>(t)].lo == lo && runs[as<usize>(t)].hi == hi) {
                    rid = t;
                    break;
                }
            }
            if (rid < 0) {
                rid = as<i32>(runs.size());
                runs.push_back(Run{lo, hi});
            }
            col_run.push_back(rid);
            col_dc.push_back(kj - half_c);
            max_k = max(max_k, hi - lo + 1);
        }

        Array<u8> vert(n, scratch.allocator());
        Array<u8> buf_p(as<usize>(h + max_k), scratch.allocator());
        Array<u8> buf_g(as<usize>(h + max_k), scratch.allocator());
        Array<u8> buf_h(as<usize>(h + max_k), scratch.allocator());

        for (i32 rid = 0; rid < as<i32>(runs.size()); ++rid) {
            const i32 lo = runs[as<usize>(rid)].lo;
            const i32 hi = runs[as<usize>(rid)].hi;
            for (i32 c = 0; c < w; ++c) {
                block_van_herk<IsErode>(src + as<usize>(c) * height,
                                        vert.data() + as<usize>(c) * height, h, lo, hi,
                                        buf_p.data(), buf_g.data(), buf_h.data());
            }
            for (i32 ci = 0; ci < as<i32>(col_run.size()); ++ci) {
                if (col_run[as<usize>(ci)] != rid) {
                    continue;
                }
                const i32 dc = col_dc[as<usize>(ci)];
                const i32 oc0 = max(0, -dc);
                const i32 oc1 = min(w, w - dc);
                for (i32 oc = oc0; oc < oc1; ++oc) {
                    u8* dcol = dst + as<usize>(oc) * height;
                    const u8* vcol = vert.data() + as<usize>(oc + dc) * height;
                    if constexpr (IsErode) {
                        for (i32 r = 0; r < h; ++r) {
                            dcol[r] = min(dcol[r], vcol[r]);
                        }
                    } else {
                        for (i32 r = 0; r < h; ++r) {
                            dcol[r] = max(dcol[r], vcol[r]);
                        }
                    }
                }
            }
        }
        return ImageStatus::kOk;
    }

    struct Off {
        i32 dr;
        i32 dc;
    };
    Array<Off> off(scratch.allocator());
    off.reserve(se_h * se_w);
    i32 dr_min = 0;
    i32 dr_max = 0;
    i32 dc_min = 0;
    i32 dc_max = 0;
    for (i32 kj = 0; kj < sw; ++kj) {
        for (i32 ki = 0; ki < sh; ++ki) {
            if (se_col_major[as<usize>(kj) * as<usize>(sh) + as<usize>(ki)] != 0) {
                const i32 dr = ki - half_r;
                const i32 dc = kj - half_c;
                off.push_back(Off{dr, dc});
                dr_min = min(dr_min, dr);
                dr_max = max(dr_max, dr);
                dc_min = min(dc_min, dc);
                dc_max = max(dc_max, dc);
            }
        }
    }
    if (off.empty()) {
        for (usize i = 0; i < n; ++i) {
            dst[i] = 0;
        }
        return ImageStatus::kOk;
    }

    const i32 ir0 = max(0, -dr_min);
    const i32 ir1 = min(h, h - dr_max);
    const i32 ic0 = max(0, -dc_min);
    const i32 ic1 = min(w, w - dc_max);
    const u8 init_val = IsErode ? u8{255} : u8{0};

    if (ir1 > ir0 && ic1 > ic0) {
        for (i32 oc = ic0; oc < ic1; ++oc) {
            u8* dcol = dst + as<usize>(oc) * height;
            for (i32 r = ir0; r < ir1; ++r) {
                dcol[r] = init_val;
            }
            for (const Off& o : off) {
                const u8* scol = src + as<usize>(oc + o.dc) * height + o.dr;
                if constexpr (IsErode) {
                    for (i32 r = ir0; r < ir1; ++r) {
                        dcol[r] = min(dcol[r], scol[r]);
                    }
                } else {
                    for (i32 r = ir0; r < ir1; ++r) {
                        dcol[r] = max(dcol[r], scol[r]);
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
            i32 best = 0;
            bool covered = false;
            for (const Off& o : off) {
                const i32 rc = oc + o.dc;
                const i32 rr = orow + o.dr;
                if (rc < 0 || rc >= w || rr < 0 || rr >= h) {
                    continue;
                }
                const i32 v = as<i32>(src[as<usize>(rc) * height + as<usize>(rr)]);
                if (!covered) {
                    best = v;
                    covered = true;
                } else if constexpr (IsErode) {
                    if (v < best) {
                        best = v;
                    }
                } else {
                    if (v > best) {
                        best = v;
                    }
                }
            }
            dst[as<usize>(oc) * height + as<usize>(orow)] = as<u8>(covered ? best : 0);
        }
    }
    return ImageStatus::kOk;
}

template <bool IsErode>
ImageStatus morph_flat_f64_impl(ArrayRef<f64> src_col_major, usize height, usize width,
                                ArrayRef<u8> se_col_major, usize se_h, usize se_w,
                                MutArrayRef<f64> dst_col_major) noexcept {
    const usize n = height * width;
    if (src_col_major.size() < n || dst_col_major.size() < n ||
        se_col_major.size() < se_h * se_w) {
        return ImageStatus::kInvalidArgument;
    }
    if (n == 0) {
        return ImageStatus::kOk;
    }

    const i32 h = as<i32>(height);
    const i32 w = as<i32>(width);
    const i32 sh = as<i32>(se_h);
    const i32 sw = as<i32>(se_w);
    const i32 half_r = sh / 2;
    const i32 half_c = sw / 2;
    const f64 qnan = Limits<f64>::quiet_nan();

    for (i32 oc = 0; oc < w; ++oc) {
        for (i32 orow = 0; orow < h; ++orow) {
            f64 best = 0.0;
            bool covered = false;
            bool any_nan = false;
            for (i32 kj = 0; kj < sw; ++kj) {
                const i32 c_in = oc + kj - half_c;
                if (c_in < 0 || c_in >= w) {
                    continue;
                }
                for (i32 ki = 0; ki < sh; ++ki) {
                    if (se_col_major[as<usize>(kj) * as<usize>(sh) + as<usize>(ki)] == 0) {
                        continue;
                    }
                    const i32 r_in = orow + ki - half_r;
                    if (r_in < 0 || r_in >= h) {
                        continue;
                    }
                    const f64 v = src_col_major[as<usize>(c_in) * height + as<usize>(r_in)];
                    if (is_nan(v)) {
                        any_nan = true;
                        continue;
                    }
                    if (!covered) {
                        best = v;
                        covered = true;
                    } else if constexpr (IsErode) {
                        if (v < best) {
                            best = v;
                        }
                    } else {
                        if (v > best) {
                            best = v;
                        }
                    }
                }
            }
            dst_col_major[as<usize>(oc) * height + as<usize>(orow)] =
                covered ? best : (any_nan ? qnan : 0.0);
        }
    }
    return ImageStatus::kOk;
}

using Lut512 = StaticArray<u8, 512>;

void bwlookup3x3_u8(const u8* src, u8* dst, usize rows, usize cols, const Lut512& lut) noexcept {
    if (rows == 0 || cols == 0) {
        return;
    }
    const isize rs = as<isize>(rows);
    const isize cs = as<isize>(cols);
    for (isize c = 0; c < cs; ++c) {
        for (isize r = 0; r < rs; ++r) {
            u32 idx = 0;
            for (isize dc = -1; dc <= 1; ++dc) {
                const isize cc = c + dc;
                if (cc < 0 || cc >= cs) {
                    continue;
                }
                const isize col_base = cc * rs;
                for (isize dr = -1; dr <= 1; ++dr) {
                    const isize rr = r + dr;
                    if (rr < 0 || rr >= rs) {
                        continue;
                    }
                    if (src[col_base + rr] != 0) {
                        const u32 k = as<u32>((dr + 1) + (dc + 1) * 3);
                        idx |= (u32{1} << k);
                    }
                }
            }
            dst[c * rs + r] = lut[idx];
        }
    }
}

void bool_and(const u8* a, const u8* b, u8* out, usize n) noexcept {
    for (usize i = 0; i < n; ++i) {
        out[i] = (a[i] != 0 && b[i] != 0) ? u8{1} : u8{0};
    }
}

void bool_not(const u8* a, u8* out, usize n) noexcept {
    for (usize i = 0; i < n; ++i) {
        out[i] = (a[i] != 0) ? u8{0} : u8{1};
    }
}

void bool_and_not(const u8* a, const u8* b, u8* out, usize n) noexcept {
    for (usize i = 0; i < n; ++i) {
        out[i] = (a[i] != 0 && b[i] == 0) ? u8{1} : u8{0};
    }
}

bool bool_equal(const u8* a, const u8* b, usize n) noexcept {
    for (usize i = 0; i < n; ++i) {
        if ((a[i] != 0) != (b[i] != 0)) {
            return false;
        }
    }
    return true;
}

bool any_set(const u8* a, usize n) noexcept {
    for (usize i = 0; i < n; ++i) {
        if (a[i] != 0) {
            return true;
        }
    }
    return false;
}

void op_open(u8* bw, usize rows, usize cols, u8* tmp) noexcept {
    bwlookup3x3_u8(bw, tmp, rows, cols, detail::luterode);
    bwlookup3x3_u8(tmp, bw, rows, cols, detail::lutdilate);
}

void op_close(u8* bw, usize rows, usize cols, u8* tmp) noexcept {
    bwlookup3x3_u8(bw, tmp, rows, cols, detail::lutdilate);
    bwlookup3x3_u8(tmp, bw, rows, cols, detail::luterode);
}

void op_shrink(u8* bw, usize rows, usize cols, u8* m, u8* cand) noexcept {
    static constexpr i32 kROff[4] = {0, 1, 1, 0};
    static constexpr i32 kCOff[4] = {0, 1, 0, 1};
    for (i32 s = 0; s < 4; ++s) {
        bwlookup3x3_u8(bw, m, rows, cols, detail::lutshrink);
        bool_and_not(bw, m, cand, rows * cols);
        const usize r_off = as<usize>(kROff[s]);
        const usize c_off = as<usize>(kCOff[s]);
        for (usize c = c_off; c < cols; c += 2) {
            const usize col_base = c * rows;
            for (usize r = r_off; r < rows; r += 2) {
                bw[col_base + r] = cand[col_base + r];
            }
        }
    }
}

void op_spur(u8* bw, usize rows, usize cols, u8* end_points, u8* e, u8* t) noexcept {
    const usize n = rows * cols;
    bool_not(bw, bw, n);
    bwlookup3x3_u8(bw, end_points, rows, cols, detail::lutspur);

    static constexpr i32 kROff[4] = {0, 1, 0, 1};
    static constexpr i32 kCOff[4] = {0, 0, 1, 1};

    for (i32 s = 0; s < 4; ++s) {
        const u8* field_src = nullptr;
        if (s == 0) {
            field_src = end_points;
        } else {
            bwlookup3x3_u8(bw, e, rows, cols, detail::lutspur);
            bool_and(e, end_points, t, n);
            field_src = t;
        }
        const usize r_off = as<usize>(kROff[s]);
        const usize c_off = as<usize>(kCOff[s]);
        for (usize c = c_off; c < cols; c += 2) {
            const usize col_base = c * rows;
            for (usize r = r_off; r < rows; r += 2) {
                const usize idx = col_base + r;
                bw[idx] = ((bw[idx] != 0) != (field_src[idx] != 0)) ? u8{1} : u8{0};
            }
        }
    }
    bool_not(bw, bw, n);
}

void op_thicken(u8* bw, usize rows, usize cols, Allocator* alloc) {
    const usize n = rows * cols;
    {
        Array<u8> iso(n, alloc);
        bwlookup3x3_u8(bw, iso.data(), rows, cols, detail::lutiso);
        if (any_set(iso.data(), n)) {
            Array<u8> grow(n, alloc);
            Array<u8> one_nbr(n, alloc);
            bwlookup3x3_u8(iso.data(), grow.data(), rows, cols, detail::lutdilate);
            bwlookup3x3_u8(bw, one_nbr.data(), rows, cols, detail::lutsingle);
            for (usize i = 0; i < n; ++i) {
                if (one_nbr[i] != 0 && grow[i] != 0) {
                    bw[i] = 1;
                }
            }
        }
    }

    const usize pr = rows + 4;
    const usize pc = cols + 4;
    const usize pn = pr * pc;
    Array<u8> c(pn, u8{1}, alloc);
    for (usize cc = 0; cc < cols; ++cc) {
        const usize src_base = cc * rows;
        const usize dst_base = (cc + 2) * pr + 2;
        for (usize rr = 0; rr < rows; ++rr) {
            c[dst_base + rr] = (bw[src_base + rr] != 0) ? u8{0} : u8{1};
        }
    }

    Array<u8> c1(pn, alloc);
    Array<u8> c2(pn, alloc);
    Array<u8> d(pn, alloc);
    bwlookup3x3_u8(c.data(), c1.data(), pr, pc, detail::lutthin1);
    bwlookup3x3_u8(c1.data(), c2.data(), pr, pc, detail::lutthin2);
    bwlookup3x3_u8(c2.data(), d.data(), pr, pc, detail::lutdiag);

    for (usize i = 0; i < pn; ++i) {
        const bool keep = (c[i] != 0) && (c2[i] == 0) && (d[i] != 0);
        c[i] = (keep || c2[i] != 0) ? u8{1} : u8{0};
    }

    for (usize cc = 0; cc < pc; ++cc) {
        const usize col_base = cc * pr;
        c[col_base + 0] = 1;
        c[col_base + 1] = 1;
        c[col_base + (pr - 2)] = 1;
        c[col_base + (pr - 1)] = 1;
    }
    for (usize rr = 0; rr < pr; ++rr) {
        c[0 * pr + rr] = 1;
        c[1 * pr + rr] = 1;
        c[(pc - 2) * pr + rr] = 1;
        c[(pc - 1) * pr + rr] = 1;
    }

    for (usize cc = 0; cc < cols; ++cc) {
        const usize src_base = (cc + 2) * pr + 2;
        const usize dst_base = cc * rows;
        for (usize rr = 0; rr < rows; ++rr) {
            bw[dst_base + rr] = (c[src_base + rr] != 0) ? u8{0} : u8{1};
        }
    }
}

void op_branchpoints(u8* bw, usize rows, usize cols, Allocator* alloc) {
    const usize n = rows * cols;
    Array<u8> cset(n, alloc);
    Array<u8> bset(n, alloc);
    bwlookup3x3_u8(bw, cset.data(), rows, cols, detail::lutbranchpoints);
    bwlookup3x3_u8(bw, bset.data(), rows, cols, detail::lutbackcount4);

    Array<u8> fc(n, alloc);
    Array<u8> vp(n, alloc);
    Array<u8> vq(n, alloc);
    for (usize i = 0; i < n; ++i) {
        const u32 b = as<u32>(bset[i]);
        const bool not_e = (b != 1u);
        fc[i] = (not_e && cset[i] != 0) ? u8{1} : u8{0};
        vp[i] = (not_e && b == 2u) ? u8{1} : u8{0};
        vq[i] = (not_e && b > 2u) ? u8{1} : u8{0};
    }

    Array<u8> dset(n, alloc);
    bwlookup3x3_u8(vq.data(), dset.data(), rows, cols, detail::lutdilate);

    for (usize i = 0; i < n; ++i) {
        const bool m = (fc[i] != 0) && (vp[i] != 0) && (dset[i] != 0);
        bw[i] = (fc[i] != 0 && !m) ? u8{1} : u8{0};
    }
}

bool run_bwmorph_pass(BwmorphOp op, u8* bw, usize rows, usize cols, u8* tmp, Allocator* alloc) {
    const usize n = rows * cols;
    const Lut512* single = nullptr;
    switch (op) {
        case BwmorphOp::kDilate: single = &detail::lutdilate; break;
        case BwmorphOp::kErode: single = &detail::luterode; break;
        case BwmorphOp::kBridge: single = &detail::lutbridge; break;
        case BwmorphOp::kClean: single = &detail::lutclean; break;
        case BwmorphOp::kDiag: single = &detail::lutdiag; break;
        case BwmorphOp::kEndpoints: single = &detail::lutendpoints; break;
        case BwmorphOp::kFatten: single = &detail::lutfatten; break;
        case BwmorphOp::kFill: single = &detail::lutfill; break;
        case BwmorphOp::kHbreak: single = &detail::luthbreak; break;
        case BwmorphOp::kMajority: single = &detail::lutmajority; break;
        case BwmorphOp::kPerim4: single = &detail::lutper4; break;
        case BwmorphOp::kPerim8: single = &detail::lutper8; break;
        case BwmorphOp::kRemove: single = &detail::lutremove; break;
        default: break;
    }
    if (single != nullptr) {
        bwlookup3x3_u8(bw, tmp, rows, cols, *single);
        for (usize i = 0; i < n; ++i) {
            bw[i] = tmp[i];
        }
        return true;
    }

    switch (op) {
        case BwmorphOp::kOpen:
            op_open(bw, rows, cols, tmp);
            return true;
        case BwmorphOp::kClose:
            op_close(bw, rows, cols, tmp);
            return true;
        case BwmorphOp::kBothat: {
            Array<u8> closed(ArrayRef<u8>(bw, n), alloc);
            Array<u8> ctmp(n, alloc);
            op_close(closed.data(), rows, cols, ctmp.data());
            bool_and_not(closed.data(), bw, bw, n);
            return true;
        }
        case BwmorphOp::kTophat: {
            Array<u8> opened(ArrayRef<u8>(bw, n), alloc);
            Array<u8> otmp(n, alloc);
            op_open(opened.data(), rows, cols, otmp.data());
            bool_and_not(bw, opened.data(), bw, n);
            return true;
        }
        case BwmorphOp::kThin:
            bwlookup3x3_u8(bw, tmp, rows, cols, detail::lutthin1);
            bwlookup3x3_u8(tmp, bw, rows, cols, detail::lutthin2);
            return true;
        case BwmorphOp::kSkel: {
            static const Lut512* const kSkel[8] = {
                &detail::lutskel1, &detail::lutskel2, &detail::lutskel3, &detail::lutskel4,
                &detail::lutskel5, &detail::lutskel6, &detail::lutskel7, &detail::lutskel8,
            };
            for (i32 k = 0; k < 8; ++k) {
                bwlookup3x3_u8(bw, tmp, rows, cols, *kSkel[k]);
                for (usize i = 0; i < n; ++i) {
                    bw[i] = tmp[i];
                }
            }
            return true;
        }
        case BwmorphOp::kShrink: {
            Array<u8> m(n, alloc);
            Array<u8> cand(n, alloc);
            op_shrink(bw, rows, cols, m.data(), cand.data());
            return true;
        }
        case BwmorphOp::kSpur: {
            Array<u8> end_points(n, alloc);
            Array<u8> e(n, alloc);
            Array<u8> t(n, alloc);
            op_spur(bw, rows, cols, end_points.data(), e.data(), t.data());
            return true;
        }
        case BwmorphOp::kThicken:
            op_thicken(bw, rows, cols, alloc);
            return true;
        case BwmorphOp::kBranchpoints:
            op_branchpoints(bw, rows, cols, alloc);
            return true;
        default:
            return false;
    }
}

} // namespace

ImageStatus strel_square(i32 n, StrelMask& out, Allocator* alloc) {
    if (n < 1) {
        n = 1;
    }
    const usize nu = as<usize>(n);
    out.rows = nu;
    out.cols = nu;
    out.mask_col_major = Array<u8>(nu * nu, u8{1}, alloc);
    return ImageStatus::kOk;
}

ImageStatus strel_rect(i32 rows, i32 cols, StrelMask& out, Allocator* alloc) {
    if (rows < 1) {
        rows = 1;
    }
    if (cols < 1) {
        cols = 1;
    }
    const usize ru = as<usize>(rows);
    const usize cu = as<usize>(cols);
    out.rows = ru;
    out.cols = cu;
    out.mask_col_major = Array<u8>(ru * cu, u8{1}, alloc);
    return ImageStatus::kOk;
}

ImageStatus strel_diamond(i32 radius, StrelMask& out, Allocator* alloc) {
    if (radius < 1) {
        radius = 1;
    }
    const i32 n = 2 * radius + 1;
    const usize nu = as<usize>(n);
    out.rows = nu;
    out.cols = nu;
    out.mask_col_major = Array<u8>(nu * nu, u8{0}, alloc);
    for (i32 c = 0; c < n; ++c) {
        for (i32 r = 0; r < n; ++r) {
            if (abs(r - radius) + abs(c - radius) <= radius) {
                out.mask_col_major[as<usize>(c) * nu + as<usize>(r)] = 1;
            }
        }
    }
    return ImageStatus::kOk;
}

ImageStatus strel_disk(f64 radius, i32 n_periodic, StrelMask& out, Allocator* alloc) {
    if (!is_finite(radius) || radius > 1e5) {
        return ImageStatus::kInvalidArgument;
    }
    if (radius < 1.0) {
        radius = 1.0;
    }
    const i32 r = as<i32>(round(radius));
    if (r < 3 || (n_periodic != 4 && n_periodic != 6 && n_periodic != 8)) {
        return strel_disk_euclidean(radius, out, alloc);
    }

    struct Vec2 {
        i32 dr;
        i32 dc;
    };
    static constexpr Vec2 kB4[4] = {{1, 0}, {1, 1}, {0, 1}, {-1, 1}};
    static constexpr Vec2 kB6[6] = {{1, 0}, {1, 2}, {2, 1}, {0, 1}, {-1, 2}, {-2, 1}};
    static constexpr Vec2 kB8[8] = {{1, 0}, {2, 1},  {1, 1},  {1, 2},
                                    {0, 1}, {-1, 2}, {-1, 1}, {-2, 1}};
    const Vec2* basis = kB4;
    i32 nb = 4;
    if (n_periodic == 6) {
        basis = kB6;
        nb = 6;
    } else if (n_periodic == 8) {
        basis = kB8;
        nb = 8;
    }

    const f64 theta = kPi / (2.0 * as<f64>(n_periodic));
    const f64 k = 2.0 * as<f64>(r) / (1.0 / tan(theta) + 1.0 / sin(theta));

    ArenaTemp scratch = get_scratch(alloc);
    GridPtSet pts(scratch.allocator());
    pts.insert(GridPt{0, 0});
    for (i32 q = 0; q < nb; ++q) {
        const i32 vr = basis[q].dr;
        const i32 vc = basis[q].dc;
        const f64 norm = hypot(as<f64>(vr), as<f64>(vc));
        const i32 rp = as<i32>(floor(k / norm));
        minksum_periodic(pts, vr, vc, rp, scratch.allocator());
    }

    i32 max_row = 0;
    for (const GridPt& p : pts) {
        max_row = max(max_row, abs(p.r));
    }
    const i32 radial_diff = r - max_row;
    const i32 len = 2 * (radial_diff - 1) + 1;
    if (len >= 3) {
        const i32 half = (len - 1) / 2;
        minksum_periodic(pts, 0, 1, half, scratch.allocator());
        minksum_periodic(pts, 1, 0, half, scratch.allocator());
    }

    i32 rad_r = 0;
    i32 rad_c = 0;
    for (const GridPt& p : pts) {
        rad_r = max(rad_r, abs(p.r));
        rad_c = max(rad_c, abs(p.c));
    }
    const usize h = as<usize>(2 * rad_r + 1);
    const usize w = as<usize>(2 * rad_c + 1);
    out.rows = h;
    out.cols = w;
    out.mask_col_major = Array<u8>(h * w, u8{0}, alloc);
    for (const GridPt& p : pts) {
        const usize pr = as<usize>(p.r + rad_r);
        const usize pc = as<usize>(p.c + rad_c);
        out.mask_col_major[pc * h + pr] = 1;
    }
    return ImageStatus::kOk;
}

ImageStatus strel_line(f64 length, f64 theta_deg, StrelMask& out, Allocator* alloc) {
    if (!is_finite(length) || !is_finite(theta_deg) || length > 1e5) {
        return ImageStatus::kInvalidArgument;
    }
    if (length < 1.0) {
        length = 1.0;
    }
    const f64 th = theta_deg * kPi / 180.0;
    const f64 cx = (length - 1.0) * 0.5 * cos(th);
    const f64 cy = (length - 1.0) * 0.5 * sin(th);
    const i32 w = max(1, as<i32>(ceil(2.0 * abs(cx) + 1.0)));
    const i32 h = max(1, as<i32>(ceil(2.0 * abs(cy) + 1.0)));
    const usize hu = as<usize>(h);
    const usize wu = as<usize>(w);
    out.rows = hu;
    out.cols = wu;
    out.mask_col_major = Array<u8>(hu * wu, u8{0}, alloc);
    const f64 r0 = as<f64>(h - 1) * 0.5;
    const f64 c0 = as<f64>(w - 1) * 0.5;
    const i32 nstep = max(1, as<i32>(ceil(length)));
    for (i32 k = 0; k < nstep; ++k) {
        const f64 t = (nstep > 1) ? (as<f64>(k) / as<f64>(nstep - 1) - 0.5) * (length - 1.0) : 0.0;
        const i32 rr = as<i32>(round(r0 + t * sin(th)));
        const i32 cc = as<i32>(round(c0 + t * cos(th)));
        if (rr >= 0 && rr < h && cc >= 0 && cc < w) {
            out.mask_col_major[as<usize>(cc) * hu + as<usize>(rr)] = 1;
        }
    }
    return ImageStatus::kOk;
}

ImageStatus morph_erode_u8(ArrayRef<u8> src_col_major, usize height, usize width,
                           ArrayRef<u8> se_col_major, usize se_h, usize se_w,
                           MutArrayRef<u8> dst_col_major, Allocator* alloc) {
    return morph_flat_u8_impl<true>(src_col_major, height, width, se_col_major, se_h, se_w,
                                    dst_col_major, alloc);
}

ImageStatus morph_dilate_u8(ArrayRef<u8> src_col_major, usize height, usize width,
                            ArrayRef<u8> se_col_major, usize se_h, usize se_w,
                            MutArrayRef<u8> dst_col_major, Allocator* alloc) {
    return morph_flat_u8_impl<false>(src_col_major, height, width, se_col_major, se_h, se_w,
                                     dst_col_major, alloc);
}

ImageStatus morph_erode_f64(ArrayRef<f64> src_col_major, usize height, usize width,
                            ArrayRef<u8> se_col_major, usize se_h, usize se_w,
                            MutArrayRef<f64> dst_col_major) noexcept {
    return morph_flat_f64_impl<true>(src_col_major, height, width, se_col_major, se_h, se_w,
                                     dst_col_major);
}

ImageStatus morph_dilate_f64(ArrayRef<f64> src_col_major, usize height, usize width,
                             ArrayRef<u8> se_col_major, usize se_h, usize se_w,
                             MutArrayRef<f64> dst_col_major) noexcept {
    return morph_flat_f64_impl<false>(src_col_major, height, width, se_col_major, se_h, se_w,
                                      dst_col_major);
}

ImageStatus imreconstruct_f64(ArrayRef<f64> marker_col_major, ArrayRef<f64> mask_col_major,
                              usize height, usize width, i32 conn, MutArrayRef<f64> out_col_major,
                              Allocator* alloc) {
    const usize n = height * width;
    if (marker_col_major.size() < n || mask_col_major.size() < n || out_col_major.size() < n) {
        return ImageStatus::kInvalidArgument;
    }
    if (n == 0) {
        return ImageStatus::kOk;
    }
    if (conn != 4) {
        conn = 8;
    }
    for (usize i = 0; i < n; ++i) {
        if (is_nan(marker_col_major[i]) || is_nan(mask_col_major[i])) {
            return ImageStatus::kInvalidArgument;
        }
    }

    ArenaTemp scratch = get_scratch(alloc);
    Array<f64> mask_copy(scratch.allocator());
    ArrayRef<f64> mask_ref = mask_col_major;
    if (out_col_major.data() == mask_col_major.data()) {
        mask_copy.resize(n);
        for (usize i = 0; i < n; ++i) {
            mask_copy[i] = mask_col_major[i];
        }
        mask_ref = mask_copy.as_ref();
    }

    for (usize i = 0; i < n; ++i) {
        out_col_major[i] = min(marker_col_major[i], mask_ref[i]);
    }

    struct Off {
        i32 dr;
        i32 dc;
    };
    static constexpr Off kOff8[8] = {{-1, -1}, {-1, 0}, {-1, 1}, {0, -1},
                                     {0, 1},   {1, -1}, {1, 0},  {1, 1}};
    static constexpr Off kOff4[4] = {{-1, 0}, {0, -1}, {0, 1}, {1, 0}};
    const Off* nbr = (conn == 8) ? kOff8 : kOff4;
    const i32 nnb = (conn == 8) ? 8 : 4;

    const i32 hi = as<i32>(height);
    const i32 wi = as<i32>(width);
    auto idx_of = [height](i32 r, i32 c) noexcept -> usize {
        return as<usize>(c) * height + as<usize>(r);
    };

    // Raster pass (causal neighbors with q < i).
    for (usize i = 0; i < n; ++i) {
        const i32 r = as<i32>(i % height);
        const i32 c = as<i32>(i / height);
        f64 m = out_col_major[i];
        for (i32 k = 0; k < nnb; ++k) {
            const i32 nr = r + nbr[k].dr;
            const i32 nc = c + nbr[k].dc;
            if (nr < 0 || nr >= hi || nc < 0 || nc >= wi) {
                continue;
            }
            const usize q = idx_of(nr, nc);
            if (q < i && out_col_major[q] > m) {
                m = out_col_major[q];
            }
        }
        out_col_major[i] = min(m, mask_ref[i]);
    }

    // Anti-raster pass (anti-causal neighbors with q > i) + FIFO seeding.
    Array<usize> fifo(scratch.allocator());
    fifo.reserve(n);
    for (usize ii = n; ii-- > 0;) {
        const usize i = ii;
        const i32 r = as<i32>(i % height);
        const i32 c = as<i32>(i / height);
        f64 m = out_col_major[i];
        for (i32 k = 0; k < nnb; ++k) {
            const i32 nr = r + nbr[k].dr;
            const i32 nc = c + nbr[k].dc;
            if (nr < 0 || nr >= hi || nc < 0 || nc >= wi) {
                continue;
            }
            const usize q = idx_of(nr, nc);
            if (q > i && out_col_major[q] > m) {
                m = out_col_major[q];
            }
        }
        out_col_major[i] = min(m, mask_ref[i]);
        for (i32 k = 0; k < nnb; ++k) {
            const i32 nr = r + nbr[k].dr;
            const i32 nc = c + nbr[k].dc;
            if (nr < 0 || nr >= hi || nc < 0 || nc >= wi) {
                continue;
            }
            const usize q = idx_of(nr, nc);
            if (q > i && out_col_major[q] < out_col_major[i] &&
                out_col_major[q] < mask_ref[q]) {
                fifo.push_back(i);
                break;
            }
        }
    }

    // FIFO propagation pass.
    usize head = 0;
    while (head < fifo.size()) {
        const usize p = fifo[head++];
        const i32 r = as<i32>(p % height);
        const i32 c = as<i32>(p / height);
        const f64 jp = out_col_major[p];
        for (i32 k = 0; k < nnb; ++k) {
            const i32 nr = r + nbr[k].dr;
            const i32 nc = c + nbr[k].dc;
            if (nr < 0 || nr >= hi || nc < 0 || nc >= wi) {
                continue;
            }
            const usize q = idx_of(nr, nc);
            if (out_col_major[q] < jp && out_col_major[q] < mask_ref[q]) {
                out_col_major[q] = min(jp, mask_ref[q]);
                fifo.push_back(q);
            }
        }
    }

    return ImageStatus::kOk;
}

ImageStatus imfill_holes_binary(ArrayRef<u8> bw_col_major, usize height, usize width, i32 conn,
                                MutArrayRef<u8> out_col_major, Allocator* alloc) {
    const usize n = height * width;
    if (bw_col_major.size() < n || out_col_major.size() < n) {
        return ImageStatus::kInvalidArgument;
    }
    if (n == 0) {
        return ImageStatus::kOk;
    }
    ArenaTemp scratch = get_scratch(alloc);
    Array<f64> marker(n, scratch.allocator());
    Array<f64> mask(n, scratch.allocator());
    Array<f64> recon(n, scratch.allocator());
    for (usize c = 0; c < width; ++c) {
        for (usize r = 0; r < height; ++r) {
            const usize idx = c * height + r;
            const bool fg = (bw_col_major[idx] != 0);
            mask[idx] = fg ? 0.0 : 1.0;
            const bool on_rim = (r == 0 || c == 0 || r + 1 == height || c + 1 == width);
            marker[idx] = (on_rim && !fg) ? 1.0 : 0.0;
        }
    }
    const ImageStatus st =
        imreconstruct_f64(marker, mask, height, width, conn, recon, scratch.allocator());
    if (st != ImageStatus::kOk) {
        return st;
    }
    for (usize i = 0; i < n; ++i) {
        out_col_major[i] = (recon[i] != 0.0) ? u8{0} : u8{1};
    }
    return ImageStatus::kOk;
}

ImageStatus imfill_holes_gray(ArrayRef<f64> img_col_major, usize height, usize width, i32 conn,
                              MutArrayRef<f64> out_col_major, Allocator* alloc) {
    const usize n = height * width;
    if (img_col_major.size() < n || out_col_major.size() < n) {
        return ImageStatus::kInvalidArgument;
    }
    if (n == 0) {
        return ImageStatus::kOk;
    }
    const f64 neg_inf = -Limits<f64>::infinity();
    ArenaTemp scratch = get_scratch(alloc);
    Array<f64> marker(n, scratch.allocator());
    Array<f64> mask(n, scratch.allocator());
    Array<f64> recon(n, scratch.allocator());
    for (usize c = 0; c < width; ++c) {
        for (usize r = 0; r < height; ++r) {
            const usize idx = c * height + r;
            const f64 ic = -img_col_major[idx];
            mask[idx] = ic;
            const bool on_rim = (r == 0 || c == 0 || r + 1 == height || c + 1 == width);
            marker[idx] = on_rim ? ic : neg_inf;
        }
    }
    const ImageStatus st =
        imreconstruct_f64(marker, mask, height, width, conn, recon, scratch.allocator());
    if (st != ImageStatus::kOk) {
        return st;
    }
    for (usize i = 0; i < n; ++i) {
        out_col_major[i] = -recon[i];
    }
    return ImageStatus::kOk;
}

ImageStatus imregionalmax_2d(ArrayRef<f64> img_col_major, usize height, usize width, i32 conn,
                             MutArrayRef<u8> out_col_major, Allocator* alloc) {
    const usize n = height * width;
    if (img_col_major.size() < n || out_col_major.size() < n) {
        return ImageStatus::kInvalidArgument;
    }
    if (n == 0) {
        return ImageStatus::kOk;
    }
    if (conn != 4) {
        conn = 8;
    }
    static constexpr i32 kDr8[8] = {-1, -1, -1, 0, 0, 1, 1, 1};
    static constexpr i32 kDc8[8] = {-1, 0, 1, -1, 1, -1, 0, 1};
    static constexpr i32 kDr4[4] = {-1, 0, 0, 1};
    static constexpr i32 kDc4[4] = {0, -1, 1, 0};
    const i32* dr = (conn == 4) ? kDr4 : kDr8;
    const i32* dc = (conn == 4) ? kDc4 : kDc8;
    const i32 nnb = (conn == 4) ? 4 : 8;
    const i32 hi = as<i32>(height);
    const i32 wi = as<i32>(width);

    ArenaTemp scratch = get_scratch(alloc);
    Array<u8> visited(n, u8{0}, scratch.allocator());
    Array<usize> comp(scratch.allocator());
    Array<usize> q(scratch.allocator());
    comp.reserve(64);
    q.reserve(64);

    for (usize i = 0; i < n; ++i) {
        out_col_major[i] = 0;
    }
    for (usize i = 0; i < n; ++i) {
        if (visited[i] != 0) {
            continue;
        }
        visited[i] = 1;
        const f64 v0 = img_col_major[i];
        if (is_nan(v0)) {
            continue;
        }
        comp.clear();
        q.clear();
        comp.push_back(i);
        q.push_back(i);
        bool has_higher = false;
        usize head = 0;
        while (head < q.size()) {
            const usize curr = q[head++];
            const i32 r = as<i32>(curr % height);
            const i32 c = as<i32>(curr / height);
            for (i32 k = 0; k < nnb; ++k) {
                const i32 nr = r + dr[k];
                const i32 nc = c + dc[k];
                if (nr < 0 || nr >= hi || nc < 0 || nc >= wi) {
                    continue;
                }
                const usize nb = as<usize>(nc) * height + as<usize>(nr);
                const f64 vn = img_col_major[nb];
                if (vn > v0) {
                    has_higher = true;
                } else if (vn == v0 && visited[nb] == 0) {
                    visited[nb] = 1;
                    comp.push_back(nb);
                    q.push_back(nb);
                }
            }
        }
        if (!has_higher) {
            for (usize j = 0; j < comp.size(); ++j) {
                out_col_major[comp[j]] = 1;
            }
        }
    }
    return ImageStatus::kOk;
}

ImageStatus imregionalmin_2d(ArrayRef<f64> img_col_major, usize height, usize width, i32 conn,
                             MutArrayRef<u8> out_col_major, Allocator* alloc) {
    const usize n = height * width;
    if (img_col_major.size() < n || out_col_major.size() < n) {
        return ImageStatus::kInvalidArgument;
    }
    if (n == 0) {
        return ImageStatus::kOk;
    }
    ArenaTemp scratch = get_scratch(alloc);
    Array<f64> inv(n, scratch.allocator());
    for (usize i = 0; i < n; ++i) {
        inv[i] = -img_col_major[i];
    }
    return imregionalmax_2d(inv, height, width, conn, out_col_major, scratch.allocator());
}

ImageStatus imhmax_2d(ArrayRef<f64> img_col_major, usize height, usize width, f64 h, i32 conn,
                      MutArrayRef<f64> out_col_major, Allocator* alloc) {
    const usize n = height * width;
    if (img_col_major.size() < n || out_col_major.size() < n || !is_finite(h) || !(h >= 0.0)) {
        return ImageStatus::kInvalidArgument;
    }
    if (n == 0) {
        return ImageStatus::kOk;
    }
    ArenaTemp scratch = get_scratch(alloc);
    Array<f64> marker(n, scratch.allocator());
    for (usize i = 0; i < n; ++i) {
        marker[i] = img_col_major[i] - h;
    }
    return imreconstruct_f64(marker, img_col_major, height, width, conn, out_col_major,
                             scratch.allocator());
}

ImageStatus imhmin_2d(ArrayRef<f64> img_col_major, usize height, usize width, f64 h, i32 conn,
                      MutArrayRef<f64> out_col_major, Allocator* alloc) {
    const usize n = height * width;
    if (img_col_major.size() < n || out_col_major.size() < n || !is_finite(h) || !(h >= 0.0)) {
        return ImageStatus::kInvalidArgument;
    }
    if (n == 0) {
        return ImageStatus::kOk;
    }
    ArenaTemp scratch = get_scratch(alloc);
    Array<f64> inv(n, scratch.allocator());
    Array<f64> j_inv(n, scratch.allocator());
    for (usize i = 0; i < n; ++i) {
        inv[i] = -img_col_major[i];
    }
    const ImageStatus st = imhmax_2d(inv, height, width, h, conn, j_inv, scratch.allocator());
    if (st != ImageStatus::kOk) {
        return st;
    }
    for (usize i = 0; i < n; ++i) {
        out_col_major[i] = -j_inv[i];
    }
    return ImageStatus::kOk;
}

ImageStatus imimposemin_2d(ArrayRef<f64> img_col_major, ArrayRef<u8> bw_col_major, usize height,
                           usize width, bool is_float_class, i32 conn,
                           MutArrayRef<f64> out_col_major, Allocator* alloc) {
    const usize n = height * width;
    if (img_col_major.size() < n || bw_col_major.size() < n || out_col_major.size() < n) {
        return ImageStatus::kInvalidArgument;
    }
    if (n == 0) {
        return ImageStatus::kOk;
    }
    const f64 pos_inf = Limits<f64>::infinity();
    const f64 neg_inf = -pos_inf;
    f64 min_i = pos_inf;
    f64 max_i = neg_inf;
    for (usize i = 0; i < n; ++i) {
        const f64 v = img_col_major[i];
        if (v < min_i) {
            min_i = v;
        }
        if (v > max_i) {
            max_i = v;
        }
    }
    if (!is_finite(min_i)) {
        min_i = 0.0;
    }
    if (!is_finite(max_i)) {
        max_i = 0.0;
    }
    const f64 h = is_float_class ? ((max_i - min_i) / 1000.0) : 1.0;
    const f64 cap_t = (is_finite(max_i) ? max_i : 0.0) + abs(h) + 1.0;

    ArenaTemp scratch = get_scratch(alloc);
    Array<f64> m_inv(n, scratch.allocator());
    Array<f64> g_inv(n, scratch.allocator());
    Array<f64> j_inv(n, scratch.allocator());
    for (usize i = 0; i < n; ++i) {
        const bool marked = (bw_col_major[i] != 0);
        const f64 md = marked ? neg_inf : (img_col_major[i] + h);
        const f64 gd = marked ? neg_inf : pos_inf;
        m_inv[i] = cap_t - md;
        g_inv[i] = cap_t - gd;
    }
    const ImageStatus st =
        imreconstruct_f64(g_inv, m_inv, height, width, conn, j_inv, scratch.allocator());
    if (st != ImageStatus::kOk) {
        return st;
    }
    for (usize i = 0; i < n; ++i) {
        out_col_major[i] = cap_t - j_inv[i];
    }
    return ImageStatus::kOk;
}

ImageStatus imclearborder_binary(ArrayRef<u8> bw_col_major, usize height, usize width, i32 conn,
                                 MutArrayRef<u8> out_col_major, Allocator* alloc) {
    const usize n = height * width;
    if (bw_col_major.size() < n || out_col_major.size() < n) {
        return ImageStatus::kInvalidArgument;
    }
    if (n == 0) {
        return ImageStatus::kOk;
    }
    ArenaTemp scratch = get_scratch(alloc);
    Array<f64> marker(n, scratch.allocator());
    Array<f64> mask(n, scratch.allocator());
    Array<f64> recon(n, scratch.allocator());
    for (usize c = 0; c < width; ++c) {
        for (usize r = 0; r < height; ++r) {
            const usize idx = c * height + r;
            const bool fg = (bw_col_major[idx] != 0);
            mask[idx] = fg ? 1.0 : 0.0;
            const bool on_rim = (r == 0 || c == 0 || r + 1 == height || c + 1 == width);
            marker[idx] = (on_rim && fg) ? 1.0 : 0.0;
        }
    }
    const ImageStatus st =
        imreconstruct_f64(marker, mask, height, width, conn, recon, scratch.allocator());
    if (st != ImageStatus::kOk) {
        return st;
    }
    for (usize i = 0; i < n; ++i) {
        const bool b = (bw_col_major[i] != 0);
        const bool rim = (recon[i] != 0.0);
        out_col_major[i] = (b && !rim) ? u8{1} : u8{0};
    }
    return ImageStatus::kOk;
}

ImageStatus imkeepborder_binary(ArrayRef<u8> bw_col_major, usize height, usize width, i32 conn,
                                MutArrayRef<u8> out_col_major, Allocator* alloc) {
    const usize n = height * width;
    if (bw_col_major.size() < n || out_col_major.size() < n) {
        return ImageStatus::kInvalidArgument;
    }
    if (n == 0) {
        return ImageStatus::kOk;
    }
    ArenaTemp scratch = get_scratch(alloc);
    Array<f64> marker(n, scratch.allocator());
    Array<f64> mask(n, scratch.allocator());
    Array<f64> recon(n, scratch.allocator());
    for (usize c = 0; c < width; ++c) {
        for (usize r = 0; r < height; ++r) {
            const usize idx = c * height + r;
            const bool fg = (bw_col_major[idx] != 0);
            mask[idx] = fg ? 1.0 : 0.0;
            const bool on_rim = (r == 0 || c == 0 || r + 1 == height || c + 1 == width);
            marker[idx] = (on_rim && fg) ? 1.0 : 0.0;
        }
    }
    const ImageStatus st =
        imreconstruct_f64(marker, mask, height, width, conn, recon, scratch.allocator());
    if (st != ImageStatus::kOk) {
        return st;
    }
    for (usize i = 0; i < n; ++i) {
        out_col_major[i] = (recon[i] != 0.0) ? u8{1} : u8{0};
    }
    return ImageStatus::kOk;
}

ImageStatus bwpack_2d(ArrayRef<u8> bw_col_major, usize height, usize width,
                      MutArrayRef<u32> out_col_major) noexcept {
    constexpr usize kBits = 32;
    const usize out_h = (height + kBits - 1) / kBits;
    if (bw_col_major.size() < height * width || out_col_major.size() < out_h * width) {
        return ImageStatus::kInvalidArgument;
    }
    if (out_h == 0 || width == 0) {
        return ImageStatus::kOk;
    }
    for (usize c = 0; c < width; ++c) {
        for (usize orow = 0; orow < out_h; ++orow) {
            u32 word = 0;
            for (usize b = 0; b < kBits; ++b) {
                const usize r = orow * kBits + b;
                if (r >= height) {
                    break;
                }
                if (bw_col_major[c * height + r] != 0) {
                    word |= (u32{1} << b);
                }
            }
            out_col_major[c * out_h + orow] = word;
        }
    }
    return ImageStatus::kOk;
}

ImageStatus bwunpack_2d(ArrayRef<u32> packed_col_major, usize packed_h, usize width, usize out_h,
                        MutArrayRef<u8> out_col_major) noexcept {
    constexpr usize kBits = 32;
    if (out_h > packed_h * kBits || packed_col_major.size() < packed_h * width ||
        out_col_major.size() < out_h * width) {
        return ImageStatus::kInvalidArgument;
    }
    if (out_h == 0 || width == 0) {
        return ImageStatus::kOk;
    }
    for (usize c = 0; c < width; ++c) {
        for (usize r = 0; r < out_h; ++r) {
            const usize orow = r / kBits;
            const usize b = r % kBits;
            const u32 bit = (packed_col_major[c * packed_h + orow] >> b) & u32{1};
            out_col_major[c * out_h + r] = as<u8>(bit);
        }
    }
    return ImageStatus::kOk;
}

ImageStatus bwlookup_nxn(ArrayRef<u8> bw_col_major, usize height, usize width, ArrayRef<f64> lut,
                         usize n, MutArrayRef<f64> out_col_major) noexcept {
    if (n == 0 || n > 5) {
        return ImageStatus::kInvalidArgument;
    }
    const usize nq = n * n;
    const usize lut_len = usize{1} << nq;
    const usize total = height * width;
    if (lut.size() < lut_len || bw_col_major.size() < total || out_col_major.size() < total) {
        return ImageStatus::kInvalidArgument;
    }
    if (total == 0) {
        return ImageStatus::kOk;
    }

    const isize hs = as<isize>(height);
    const isize ws = as<isize>(width);
    const isize ns = as<isize>(n);
    const isize half = as<isize>((n - 1) / 2);

    for (isize c = 0; c < ws; ++c) {
        for (isize r = 0; r < hs; ++r) {
            usize idx = 0;
            for (isize v = 0; v < ns; ++v) {
                const isize cc = c + v - half;
                if (cc < 0 || cc >= ws) {
                    continue;
                }
                const usize col_base = as<usize>(cc) * height;
                for (isize u = 0; u < ns; ++u) {
                    const isize rr = r + u - half;
                    if (rr < 0 || rr >= hs) {
                        continue;
                    }
                    if (bw_col_major[col_base + as<usize>(rr)] != 0) {
                        const usize bit_pos = (nq - 1) - (as<usize>(u) + n * as<usize>(v));
                        idx |= (usize{1} << bit_pos);
                    }
                }
            }
            out_col_major[as<usize>(c) * height + as<usize>(r)] = lut[idx];
        }
    }
    return ImageStatus::kOk;
}

ImageStatus bwmorph_2d(ArrayRef<u8> bw_col_major, usize height, usize width, BwmorphOp op,
                       i32 iterations, MutArrayRef<u8> out_col_major, Allocator* alloc) {
    const usize n = height * width;
    if (bw_col_major.size() < n || out_col_major.size() < n) {
        return ImageStatus::kInvalidArgument;
    }
    for (usize i = 0; i < n; ++i) {
        out_col_major[i] = (bw_col_major[i] != 0) ? u8{1} : u8{0};
    }
    if (n == 0 || iterations == 0) {
        return ImageStatus::kOk;
    }

    constexpr i32 kSafetyCap = 10000;
    const i32 max_passes = (iterations < 0) ? kSafetyCap : iterations;

    ArenaTemp scratch = get_scratch(alloc);
    Array<u8> prev(n, scratch.allocator());
    Array<u8> tmp(n, scratch.allocator());

    for (i32 pass = 0; pass < max_passes; ++pass) {
        for (usize i = 0; i < n; ++i) {
            prev[i] = out_col_major[i];
        }
        ArenaTemp pass_scratch = get_scratch(alloc);
        if (!run_bwmorph_pass(op, out_col_major.data(), height, width, tmp.data(),
                              pass_scratch.allocator())) {
            return ImageStatus::kInvalidArgument;
        }
        if (bool_equal(out_col_major.data(), prev.data(), n)) {
            break;
        }
    }
    return ImageStatus::kOk;
}

ImageStatus bwmorph3_3d(ArrayRef<u8> vol_col_major, usize height, usize width, usize pages,
                        Bwmorph3Op op, MutArrayRef<u8> out_col_major) noexcept {
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
    const i32 p_cnt = as<i32>(pages);
    auto at = [&](i32 r, i32 c, i32 p) noexcept -> i32 {
        if (r < 0 || r >= h || c < 0 || c >= w || p < 0 || p >= p_cnt) {
            return 0;
        }
        return vol_col_major[as<usize>(p) * plane + as<usize>(c) * height + as<usize>(r)] != 0 ? 1
                                                                                               : 0;
    };

    for (i32 p = 0; p < p_cnt; ++p) {
        for (i32 c = 0; c < w; ++c) {
            for (i32 r = 0; r < h; ++r) {
                const i32 centre = at(r, c, p);
                i32 count = 0;
                for (i32 dp = -1; dp <= 1; ++dp) {
                    for (i32 dc = -1; dc <= 1; ++dc) {
                        for (i32 dr = -1; dr <= 1; ++dr) {
                            count += at(r + dr, c + dc, p + dp);
                        }
                    }
                }
                const i32 faces6 = at(r - 1, c, p) + at(r + 1, c, p) + at(r, c - 1, p) +
                                   at(r, c + 1, p) + at(r, c, p - 1) + at(r, c, p + 1);
                bool v = false;
                switch (op) {
                    case Bwmorph3Op::kBranchpoints: v = (centre != 0 && count > 3); break;
                    case Bwmorph3Op::kClean: v = (centre != 0 && count != 1); break;
                    case Bwmorph3Op::kEndpoints: v = (centre != 0 && count == 2); break;
                    case Bwmorph3Op::kFill: v = (centre != 0 || faces6 == 6); break;
                    case Bwmorph3Op::kMajority: v = (count > 13); break;
                    case Bwmorph3Op::kRemove: v = (centre != 0 && faces6 != 6); break;
                }
                const usize idx = as<usize>(p) * plane + as<usize>(c) * height + as<usize>(r);
                out_col_major[idx] = v ? u8{1} : u8{0};
            }
        }
    }
    return ImageStatus::kOk;
}

ImageStatus bwtraceboundary_2d(ArrayRef<u8> bw_col_major, usize height, usize width, i64 start_row,
                               i64 start_col, i32 first_step_dir, i32 conn, usize max_points,
                               bool clockwise, Array<BoundaryPoint>& out_points) {
    const usize total = height * width;
    if (bw_col_major.size() < total || (conn != 4 && conn != 8)) {
        return ImageStatus::kInvalidArgument;
    }
    if (start_row < 0 || start_col < 0 || as<usize>(start_row) >= height ||
        as<usize>(start_col) >= width) {
        return ImageStatus::kInvalidArgument;
    }
    if (first_step_dir < 0 || first_step_dir >= conn) {
        return ImageStatus::kInvalidArgument;
    }
    if (bw_col_major[as<usize>(start_col) * height + as<usize>(start_row)] == 0) {
        return ImageStatus::kInvalidArgument;
    }

    static constexpr i32 kDr8[8] = {-1, -1, 0, 1, 1, 1, 0, -1};
    static constexpr i32 kDc8[8] = {0, 1, 1, 1, 0, -1, -1, -1};
    static constexpr i32 kDr4[4] = {-1, 0, 1, 0};
    static constexpr i32 kDc4[4] = {0, 1, 0, -1};

    const i32 nd = conn;
    const i32* dr_tab = (conn == 8) ? kDr8 : kDr4;
    const i32* dc_tab = (conn == 8) ? kDc8 : kDc4;
    const i32 rot = clockwise ? 1 : -1;

    auto fg_at = [&](i64 r, i64 c) noexcept -> bool {
        if (r < 0 || c < 0 || as<usize>(r) >= height || as<usize>(c) >= width) {
            return false;
        }
        return bw_col_major[as<usize>(c) * height + as<usize>(r)] != 0;
    };

    out_points.clear();
    out_points.reserve(64);
    out_points.push_back(BoundaryPoint{start_row, start_col});

    i32 back_dir = (first_step_dir + nd / 2) % nd;
    i64 cur_r = start_row;
    i64 cur_c = start_col;
    const usize cap = (max_points == 0) ? Limits<usize>::kMax : max_points;

    while (out_points.size() < cap) {
        bool found = false;
        for (i32 k = 0; k < nd; ++k) {
            const i32 d = (back_dir + rot * (k + 1) + nd * 8) % nd;
            const i64 nr = cur_r + as<i64>(dr_tab[d]);
            const i64 nc = cur_c + as<i64>(dc_tab[d]);
            if (fg_at(nr, nc)) {
                out_points.push_back(BoundaryPoint{nr, nc});
                cur_r = nr;
                cur_c = nc;
                back_dir = (d + nd / 2) % nd;
                found = true;
                break;
            }
        }
        if (!found) {
            out_points.push_back(BoundaryPoint{start_row, start_col});
            break;
        }
        if (cur_r == start_row && cur_c == start_col && out_points.size() >= 2) {
            break;
        }
    }
    return ImageStatus::kOk;
}

} // namespace keel::image
