#include "keel/image.h"

#include "keel/arena.h"
#include "keel/array.h"
#include "keel/math.h"

namespace keel::image {

namespace {

inline usize lin3(usize r, usize c, usize p, usize h, usize w) noexcept {
    return r + c * h + p * h * w;
}

} // namespace

ImageStatus fspecial3_average(usize h, usize w, usize p, MutArrayRef<f64> out_col_major) noexcept {
    const usize n = h * w * p;
    if (n == 0u || out_col_major.size() < n) return ImageStatus::kInvalidArgument;
    const f64 v = 1.0 / as<f64>(n);
    for (usize i = 0; i < n; ++i) out_col_major[i] = v;
    return ImageStatus::kOk;
}

ImageStatus fspecial3_gaussian(usize h, usize w, usize p, f64 sr, f64 sc, f64 sp,
                               MutArrayRef<f64> out_col_major) noexcept {
    const usize n = h * w * p;
    if (n == 0u || out_col_major.size() < n || !(sr > 0.0) || !(sc > 0.0) || !(sp > 0.0)) {
        return ImageStatus::kInvalidArgument;
    }
    const f64 cy = 0.5 * (as<f64>(h) - 1.0);
    const f64 cx = 0.5 * (as<f64>(w) - 1.0);
    const f64 cz = 0.5 * (as<f64>(p) - 1.0);
    f64 sum = 0.0;
    for (usize z = 0; z < p; ++z) {
        const f64 dp = (as<f64>(z) - cz) / sp;
        for (usize c = 0; c < w; ++c) {
            const f64 dc = (as<f64>(c) - cx) / sc;
            for (usize r = 0; r < h; ++r) {
                const f64 dr = (as<f64>(r) - cy) / sr;
                const f64 v = keel::exp(-0.5 * (dr * dr + dc * dc + dp * dp));
                out_col_major[lin3(r, c, z, h, w)] = v;
                sum += v;
            }
        }
    }
    if (sum > 0.0) {
        const f64 inv = 1.0 / sum;
        for (usize i = 0; i < n; ++i) out_col_major[i] *= inv;
    }
    return ImageStatus::kOk;
}

ImageStatus fspecial3_ellipsoid(f64 ar, f64 ac, f64 ap, Array<f64>& out_col_major, usize& out_h,
                                usize& out_w, usize& out_p, Allocator* alloc) {
    if (!(ar > 0.0) || !(ac > 0.0) || !(ap > 0.0)) return ImageStatus::kInvalidArgument;
    out_h = as<usize>(2.0 * keel::ceil(ar) + 1.0);
    out_w = as<usize>(2.0 * keel::ceil(ac) + 1.0);
    out_p = as<usize>(2.0 * keel::ceil(ap) + 1.0);
    const usize n = out_h * out_w * out_p;
    out_col_major = Array<f64>(n, 0.0, alloc);

    const f64 cy = 0.5 * (as<f64>(out_h) - 1.0);
    const f64 cx = 0.5 * (as<f64>(out_w) - 1.0);
    const f64 cz = 0.5 * (as<f64>(out_p) - 1.0);
    f64 sum = 0.0;
    for (usize z = 0; z < out_p; ++z) {
        const f64 dp = (as<f64>(z) - cz) / ap;
        for (usize c = 0; c < out_w; ++c) {
            const f64 dc = (as<f64>(c) - cx) / ac;
            for (usize r = 0; r < out_h; ++r) {
                const f64 dr = (as<f64>(r) - cy) / ar;
                const f64 v = (dr * dr + dc * dc + dp * dp <= 1.0) ? 1.0 : 0.0;
                out_col_major[lin3(r, c, z, out_h, out_w)] = v;
                sum += v;
            }
        }
    }
    if (sum > 0.0) {
        const f64 inv = 1.0 / sum;
        for (usize i = 0; i < n; ++i) out_col_major[i] *= inv;
    }
    return ImageStatus::kOk;
}

ImageStatus fspecial3_laplacian(f64 gamma1, f64 gamma2,
                                MutArrayRef<f64> out_col_major_3x3x3) noexcept {
    if (gamma1 < 0.0 || gamma2 < 0.0 || gamma1 + gamma2 > 1.0 || out_col_major_3x3x3.size() < 27u) {
        return ImageStatus::kInvalidArgument;
    }
    const f64 face = 1.0 - gamma1 - gamma2;
    const f64 edge = gamma1 * 0.25;
    const f64 corner = gamma2 * 0.25;
    const f64 center = -6.0 + 3.0 * gamma1 + 4.0 * gamma2;

    for (usize z = 0; z < 3u; ++z) {
        const i32 dp = as<i32>(z) - 1;
        for (usize c = 0; c < 3u; ++c) {
            const i32 dc = as<i32>(c) - 1;
            for (usize r = 0; r < 3u; ++r) {
                const i32 dr = as<i32>(r) - 1;
                const i32 d2 = dr * dr + dc * dc + dp * dp;
                f64 val = corner;
                if (d2 == 0) val = center;
                else if (d2 == 1) val = face;
                else if (d2 == 2) val = edge;
                out_col_major_3x3x3[lin3(r, c, z, 3u, 3u)] = val;
            }
        }
    }
    return ImageStatus::kOk;
}

ImageStatus fspecial3_log(usize h, usize w, usize p, f64 sr, f64 sc, f64 sp,
                          MutArrayRef<f64> out_col_major, Allocator* alloc) {
    const usize n = h * w * p;
    if (n == 0u || out_col_major.size() < n || !(sr > 0.0) || !(sc > 0.0) || !(sp > 0.0)) {
        return ImageStatus::kInvalidArgument;
    }
    ArenaTemp scratch = get_scratch(alloc);
    Array<f64> g(n, 0.0, scratch.allocator());
    const ImageStatus st = fspecial3_gaussian(h, w, p, sr, sc, sp, g.as_mut_ref());
    if (st != ImageStatus::kOk) return st;

    const f64 cy = 0.5 * (as<f64>(h) - 1.0);
    const f64 cx = 0.5 * (as<f64>(w) - 1.0);
    const f64 cz = 0.5 * (as<f64>(p) - 1.0);
    const f64 sr2 = sr * sr;
    const f64 sc2 = sc * sc;
    const f64 sp2 = sp * sp;

    f64 mean = 0.0;
    for (usize z = 0; z < p; ++z) {
        const f64 dp = as<f64>(z) - cz;
        for (usize c = 0; c < w; ++c) {
            const f64 dc = as<f64>(c) - cx;
            for (usize r = 0; r < h; ++r) {
                const f64 dr = as<f64>(r) - cy;
                const usize idx = lin3(r, c, z, h, w);
                const f64 lap = (dr * dr - sr2) / (sr2 * sr2) + (dc * dc - sc2) / (sc2 * sc2) +
                                (dp * dp - sp2) / (sp2 * sp2);
                const f64 val = g[idx] * lap;
                out_col_major[idx] = val;
                mean += val;
            }
        }
    }
    mean /= as<f64>(n);
    for (usize i = 0; i < n; ++i) out_col_major[i] -= mean;
    return ImageStatus::kOk;
}

ImageStatus fspecial3_gradient(bool sobel, Axis3D axis,
                               MutArrayRef<f64> out_col_major_3x3x3) noexcept {
    if (out_col_major_3x3x3.size() < 27u) return ImageStatus::kInvalidArgument;
    const f64 w1 = sobel ? 2.0 : 1.0;
    auto sm = [&](i32 x) noexcept -> f64 { return (x == 0) ? w1 : 1.0; };

    for (usize z = 0; z < 3u; ++z) {
        const i32 dp = as<i32>(z) - 1;
        for (usize c = 0; c < 3u; ++c) {
            const i32 dc = as<i32>(c) - 1;
            for (usize r = 0; r < 3u; ++r) {
                const i32 dr = as<i32>(r) - 1;
                f64 val = 0.0;
                switch (axis) {
                    case Axis3D::kX: val = as<f64>(-dc) * sm(dr) * sm(dp); break;
                    case Axis3D::kY: val = as<f64>(-dr) * sm(dc) * sm(dp); break;
                    case Axis3D::kZ: val = as<f64>(-dp) * sm(dr) * sm(dc); break;
                }
                out_col_major_3x3x3[lin3(r, c, z, 3u, 3u)] = val;
            }
        }
    }
    return ImageStatus::kOk;
}

ImageStatus fwind2(ArrayRef<f64> hd_col_major, ArrayRef<f64> win_col_major, usize height,
                   usize width, MutArrayRef<f64> out_col_major) noexcept {
    const usize n = height * width;
    if (hd_col_major.size() < n || win_col_major.size() < n || out_col_major.size() < n) {
        return ImageStatus::kInvalidArgument;
    }
    if (n == 0u) return ImageStatus::kOk;

    const f64 cm = 0.5 * (as<f64>(height) - 1.0);
    const f64 cn = 0.5 * (as<f64>(width) - 1.0);
    const f64 inv = 1.0 / as<f64>(n);
    const f64 fh = as<f64>(height);
    const f64 fw = as<f64>(width);

    for (usize mi = 0; mi < height; ++mi) {
        const f64 m = as<f64>(mi) - cm;
        for (usize ni = 0; ni < width; ++ni) {
            const f64 nv = as<f64>(ni) - cn;
            f64 acc_re = 0.0;
            for (usize p = 0; p < height; ++p) {
                const f64 fp = as<f64>(p) - cm;
                for (usize q = 0; q < width; ++q) {
                    const f64 fq = as<f64>(q) - cn;
                    const f64 phase = kTau * (m * fp / fh + nv * fq / fw);
                    acc_re += hd_col_major[p + q * height] * keel::cos(phase);
                }
            }
            out_col_major[mi + ni * height] = acc_re * inv * win_col_major[mi + ni * height];
        }
    }

    f64 sum = 0.0;
    for (usize i = 0; i < n; ++i) sum += out_col_major[i];
    if (sum != 0.0) {
        const f64 scale = 1.0 / sum;
        for (usize i = 0; i < n; ++i) out_col_major[i] *= scale;
    }
    return ImageStatus::kOk;
}

} // namespace keel::image
