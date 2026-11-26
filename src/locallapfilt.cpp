#include "keel/image.h"

#include "keel/arena.h"
#include "keel/array.h"
#include "keel/math.h"

namespace keel::image {

namespace {

constexpr f32 kK0 = 1.0f / 16.0f;
constexpr f32 kK1 = 4.0f / 16.0f;
constexpr f32 kK2 = 6.0f / 16.0f;
constexpr f32 kK3 = 4.0f / 16.0f;
constexpr f32 kK4 = 1.0f / 16.0f;

constexpr f32 kU0 = 2.0f / 16.0f;
constexpr f32 kU1 = 8.0f / 16.0f;
constexpr f32 kU2 = 12.0f / 16.0f;
constexpr f32 kU3 = 8.0f / 16.0f;
constexpr f32 kU4 = 2.0f / 16.0f;

struct Plane {
    usize h = 0u;
    usize w = 0u;
    Array<f32> v;

    Plane() = default;
    Plane(usize h_, usize w_, Allocator* alloc) : h(h_), w(w_), v(h_ * w_, 0.0f, alloc) {}

    f32& at(usize r, usize c) noexcept { return v[c * h + r]; }
    f32 at(usize r, usize c) const noexcept { return v[c * h + r]; }
};

inline usize reflect_idx(i64 i, usize n) noexcept {
    if (n == 0u) return 0u;
    const i64 ni = as<i64>(n);
    const i64 period = 2 * ni;
    i64 m = (i < 0) ? ((-i - 1) % period) : (i % period);
    if (m >= ni) m = period - 1 - m;
    return as<usize>(m);
}

Plane pyr_downsample(const Plane& p, Allocator* alloc) {
    const usize h = p.h;
    const usize w = p.w;
    if (h == 0u || w == 0u) return Plane(0u, 0u, alloc);

    Plane tmp(h, w, alloc);
    for (usize c = 0; c < w; ++c) {
        for (usize r = 0; r < h; ++r) {
            const i64 ri = as<i64>(r);
            tmp.at(r, c) = kK0 * p.at(reflect_idx(ri - 2, h), c) +
                           kK1 * p.at(reflect_idx(ri - 1, h), c) + kK2 * p.at(r, c) +
                           kK3 * p.at(reflect_idx(ri + 1, h), c) +
                           kK4 * p.at(reflect_idx(ri + 2, h), c);
        }
    }
    Plane sm(h, w, alloc);
    for (usize c = 0; c < w; ++c) {
        const i64 ci = as<i64>(c);
        const usize cm2 = reflect_idx(ci - 2, w);
        const usize cm1 = reflect_idx(ci - 1, w);
        const usize cp1 = reflect_idx(ci + 1, w);
        const usize cp2 = reflect_idx(ci + 2, w);
        for (usize r = 0; r < h; ++r) {
            sm.at(r, c) = kK0 * tmp.at(r, cm2) + kK1 * tmp.at(r, cm1) + kK2 * tmp.at(r, c) +
                          kK3 * tmp.at(r, cp1) + kK4 * tmp.at(r, cp2);
        }
    }

    const usize h2 = (h + 1u) / 2u;
    const usize w2 = (w + 1u) / 2u;
    Plane out(h2, w2, alloc);
    for (usize c = 0; c < w2; ++c) {
        for (usize r = 0; r < h2; ++r) {
            out.at(r, c) = sm.at(r * 2u, c * 2u);
        }
    }
    return out;
}

Plane pyr_upsample(const Plane& p, usize h_target, usize w_target, Allocator* alloc) {
    if (h_target == 0u || w_target == 0u) return Plane(0u, 0u, alloc);

    Plane up(h_target, w_target, alloc);
    for (usize c = 0; c < p.w; ++c) {
        const usize cu = 2u * c;
        if (cu >= w_target) break;
        for (usize r = 0; r < p.h; ++r) {
            const usize ru = 2u * r;
            if (ru >= h_target) break;
            up.at(ru, cu) = p.at(r, c);
        }
    }

    Plane tmp(h_target, w_target, alloc);
    for (usize c = 0; c < w_target; ++c) {
        for (usize r = 0; r < h_target; ++r) {
            const i64 ri = as<i64>(r);
            tmp.at(r, c) = kU0 * up.at(reflect_idx(ri - 2, h_target), c) +
                           kU1 * up.at(reflect_idx(ri - 1, h_target), c) + kU2 * up.at(r, c) +
                           kU3 * up.at(reflect_idx(ri + 1, h_target), c) +
                           kU4 * up.at(reflect_idx(ri + 2, h_target), c);
        }
    }

    Plane out(h_target, w_target, alloc);
    for (usize c = 0; c < w_target; ++c) {
        const i64 ci = as<i64>(c);
        const usize cm2 = reflect_idx(ci - 2, w_target);
        const usize cm1 = reflect_idx(ci - 1, w_target);
        const usize cp1 = reflect_idx(ci + 1, w_target);
        const usize cp2 = reflect_idx(ci + 2, w_target);
        for (usize r = 0; r < h_target; ++r) {
            out.at(r, c) = kU0 * tmp.at(r, cm2) + kU1 * tmp.at(r, cm1) + kU2 * tmp.at(r, c) +
                           kU3 * tmp.at(r, cp1) + kU4 * tmp.at(r, cp2);
        }
    }
    return out;
}

Plane remap_plane(const Plane& in, f32 g, f32 sigma, f32 alpha, f32 beta, Allocator* alloc) {
    Plane out(in.h, in.w, alloc);
    const f32 inv_sigma = (sigma > 0.0f) ? (1.0f / sigma) : 0.0f;
    const usize n = in.h * in.w;
    for (usize i = 0; i < n; ++i) {
        const f32 d = in.v[i] - g;
        const f32 ad = keel::abs(d);
        const f32 sgn = (d > 0.0f) ? 1.0f : ((d < 0.0f) ? -1.0f : 0.0f);
        f32 nd = 0.0f;
        if (sigma == 0.0f) {
            nd = sgn * beta * ad;
        } else if (ad <= sigma) {
            nd = sgn * sigma * as<f32>(keel::pow(as<f64>(ad * inv_sigma), as<f64>(alpha)));
        } else {
            nd = sgn * (sigma + beta * (ad - sigma));
        }
        out.v[i] = g + nd;
    }
    return out;
}

i32 auto_intensity_levels(f32 alpha) noexcept {
    const f64 a = as<f64>(alpha);
    if (a < 0.1) return 50;
    if (a < 0.9) {
        const f64 num = (50.0 * 0.9 - 16.0 * 0.1) - (50.0 - 16.0) * a;
        return as<i32>(keel::round(num / 0.8));
    }
    return 16;
}

Plane llf_core(const Plane& input, f32 sigma, f32 alpha, f32 beta, i32 num_intensity_levels,
               Allocator* alloc) {
    if ((alpha == 1.0f && beta == 1.0f) || (sigma == 0.0f && beta == 1.0f)) {
        Plane copy(input.h, input.w, alloc);
        for (usize i = 0; i < input.v.size(); ++i) copy.v[i] = input.v[i];
        return copy;
    }

    const i32 n_levels = (num_intensity_levels > 0) ? num_intensity_levels : auto_intensity_levels(alpha);

    usize short_dim = keel::min(input.h, input.w);
    i32 num_pyr = 0;
    while (short_dim >= 1u) {
        ++num_pyr;
        short_dim >>= 1u;
    }
    if (num_pyr < 1) num_pyr = 1;

    f32 min_val = Limits<f32>::infinity();
    f32 max_val = -Limits<f32>::infinity();
    for (usize i = 0; i < input.v.size(); ++i) {
        const f32 val = input.v[i];
        if (!keel::is_finite(as<f64>(val))) continue;
        if (val < min_val) min_val = val;
        if (val > max_val) max_val = val;
    }

    if (!keel::is_finite(as<f64>(min_val)) || !keel::is_finite(as<f64>(max_val)) ||
        min_val > max_val) {
        Plane copy(input.h, input.w, alloc);
        for (usize i = 0; i < input.v.size(); ++i) copy.v[i] = input.v[i];
        return copy;
    }
    if (n_levels == 1) {
        return remap_plane(input, 0.5f * (min_val + max_val), sigma, alpha, beta, alloc);
    }
    if (min_val == max_val || !keel::is_finite(as<f64>(max_val - min_val))) {
        Plane copy(input.h, input.w, alloc);
        for (usize i = 0; i < input.v.size(); ++i) copy.v[i] = input.v[i];
        return copy;
    }

    const usize np = as<usize>(num_pyr);
    Array<Plane> in_g(alloc);
    in_g.reserve(np);
    {
        Plane base(input.h, input.w, alloc);
        for (usize i = 0; i < input.v.size(); ++i) base.v[i] = input.v[i];
        in_g.push_back(keel::move(base));
    }
    for (usize i = 1; i < np; ++i) {
        in_g.push_back(pyr_downsample(in_g[i - 1u], alloc));
    }

    Array<Plane> out_l(alloc);
    out_l.reserve(np);
    for (usize i = 0; i + 1u < np; ++i) {
        out_l.push_back(Plane(in_g[i].h, in_g[i].w, alloc));
    }
    {
        Plane top(in_g[np - 1u].h, in_g[np - 1u].w, alloc);
        for (usize i = 0; i < top.v.size(); ++i) top.v[i] = in_g[np - 1u].v[i];
        out_l.push_back(keel::move(top));
    }

    const f32 delta = (max_val - min_val) / as<f32>(n_levels - 1);
    Array<Plane> r_g(alloc);
    r_g.reserve(np);
    for (usize i = 0; i < np; ++i) r_g.push_back(Plane{});

    for (i32 k = 0; k < n_levels; ++k) {
        const f32 ref_val = min_val + as<f32>(k) * delta;
        r_g[0] = remap_plane(input, ref_val, sigma, alpha, beta, alloc);
        for (usize i = 1; i < np; ++i) {
            r_g[i] = pyr_downsample(r_g[i - 1u], alloc);
        }
        for (i32 i = num_pyr - 2; i >= 0; --i) {
            const usize ui = as<usize>(i);
            const Plane up = pyr_upsample(r_g[ui + 1u], r_g[ui].h, r_g[ui].w, alloc);
            for (usize j = 0; j < out_l[ui].v.size(); ++j) {
                const f32 r_l = r_g[ui].v[j] - up.v[j];
                const f32 w = 1.0f - keel::abs(in_g[ui].v[j] - ref_val) / delta;
                if (w > 0.0f) out_l[ui].v[j] += w * r_l;
            }
        }
    }

    for (i32 i = num_pyr - 2; i >= 0; --i) {
        const usize ui = as<usize>(i);
        const Plane up = pyr_upsample(out_l[ui + 1u], out_l[ui].h, out_l[ui].w, alloc);
        for (usize j = 0; j < out_l[ui].v.size(); ++j) {
            out_l[ui].v[j] += up.v[j];
        }
    }
    return keel::move(out_l[0]);
}

} // namespace

ImageStatus local_laplacian_gray(ArrayRef<f32> in_col_major, usize height, usize width, f32 sigma,
                                 f32 alpha, f32 beta, i32 num_intensity_levels,
                                 MutArrayRef<f32> out_col_major, Allocator* alloc) {
    if (height == 0u || width == 0u || height > Limits<usize>::kMax / width ||
        !keel::is_finite(as<f64>(sigma)) || sigma < 0.0f ||
        !keel::is_finite(as<f64>(alpha)) || !(alpha > 0.0f) || !keel::is_finite(as<f64>(beta)) ||
        beta < 0.0f) {
        return ImageStatus::kInvalidArgument;
    }
    const usize n = height * width;
    if (in_col_major.size() < n || out_col_major.size() < n) {
        return ImageStatus::kInvalidArgument;
    }

    ArenaTemp scratch = get_scratch(alloc);
    Plane in_p(height, width, scratch.allocator());
    for (usize i = 0; i < n; ++i) in_p.v[i] = in_col_major[i];

    const Plane res = llf_core(in_p, sigma, alpha, beta, num_intensity_levels, scratch.allocator());
    for (usize i = 0; i < n; ++i) out_col_major[i] = res.v[i];
    return ImageStatus::kOk;
}

ImageStatus local_laplacian_rgb(ArrayRef<f32> in_r, ArrayRef<f32> in_g, ArrayRef<f32> in_b,
                                usize height, usize width, f32 sigma, f32 alpha, f32 beta,
                                i32 num_intensity_levels, bool process_luminance,
                                MutArrayRef<f32> out_r, MutArrayRef<f32> out_g,
                                MutArrayRef<f32> out_b, Allocator* alloc) {
    if (height == 0u || width == 0u || height > Limits<usize>::kMax / width ||
        !keel::is_finite(as<f64>(sigma)) || sigma < 0.0f ||
        !keel::is_finite(as<f64>(alpha)) || !(alpha > 0.0f) || !keel::is_finite(as<f64>(beta)) ||
        beta < 0.0f) {
        return ImageStatus::kInvalidArgument;
    }
    const usize n = height * width;
    if (in_r.size() < n || in_g.size() < n || in_b.size() < n || out_r.size() < n ||
        out_g.size() < n || out_b.size() < n) {
        return ImageStatus::kInvalidArgument;
    }

    if ((alpha == 1.0f && beta == 1.0f) || (sigma == 0.0f && beta == 1.0f)) {
        for (usize i = 0; i < n; ++i) {
            out_r[i] = in_r[i];
            out_g[i] = in_g[i];
            out_b[i] = in_b[i];
        }
        return ImageStatus::kOk;
    }

    ArenaTemp scratch = get_scratch(alloc);
    if (process_luminance) {
        constexpr f32 kRC = 0.298936021293776f;
        constexpr f32 kGC = 0.587043074451121f;
        constexpr f32 kBC = 0.114020904255103f;
        constexpr f32 kEps = Limits<f32>::kEpsilon;

        Plane gray(height, width, scratch.allocator());
        Array<f32> rr(n, 0.0f, scratch.allocator());
        Array<f32> rg(n, 0.0f, scratch.allocator());
        Array<f32> rb(n, 0.0f, scratch.allocator());
        for (usize i = 0; i < n; ++i) {
            const f32 y = kRC * in_r[i] + kGC * in_g[i] + kBC * in_b[i];
            gray.v[i] = y;
            const f32 yp = y + kEps;
            rr[i] = in_r[i] / yp;
            rg[i] = in_g[i] / yp;
            rb[i] = in_b[i] / yp;
        }
        const Plane filt =
            llf_core(gray, sigma, alpha, beta, num_intensity_levels, scratch.allocator());
        for (usize i = 0; i < n; ++i) {
            out_r[i] = filt.v[i] * rr[i];
            out_g[i] = filt.v[i] * rg[i];
            out_b[i] = filt.v[i] * rb[i];
        }
        return ImageStatus::kOk;
    }

    const ImageStatus st_r = local_laplacian_gray(in_r, height, width, sigma, alpha, beta,
                                                  num_intensity_levels, out_r, scratch.allocator());
    if (st_r != ImageStatus::kOk) return st_r;
    const ImageStatus st_g = local_laplacian_gray(in_g, height, width, sigma, alpha, beta,
                                                  num_intensity_levels, out_g, scratch.allocator());
    if (st_g != ImageStatus::kOk) return st_g;
    return local_laplacian_gray(in_b, height, width, sigma, alpha, beta, num_intensity_levels,
                                out_b, scratch.allocator());
}

} // namespace keel::image
