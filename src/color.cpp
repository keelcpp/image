#include "keel/image.h"

#include "keel/math.h"

namespace keel::image {

namespace {

constexpr f64 kLumaR = 0.298936021293775;
constexpr f64 kLumaG = 0.587043074451121;
constexpr f64 kLumaB = 0.114020904255103;

constexpr f64 kXyzXn = 0.95047;
constexpr f64 kXyzYn = 1.00000;
constexpr f64 kXyzZn = 1.08883;

[[nodiscard]] f64 f_lab(f64 t) noexcept {
    constexpr f64 delta = 6.0 / 29.0;
    constexpr f64 delta3 = delta * delta * delta;
    if (t > delta3) {
        return keel::cbrt(t);
    }
    return t / (3.0 * delta * delta) + 4.0 / 29.0;
}

[[nodiscard]] f64 finv_lab(f64 t) noexcept {
    constexpr f64 delta = 6.0 / 29.0;
    if (t > delta) {
        return t * t * t;
    }
    return 3.0 * delta * delta * (t - 4.0 / 29.0);
}

[[nodiscard]] f64 signed_srgb_encode(f64 c) noexcept {
    return (c >= 0.0) ? linear_to_srgb(c) : -linear_to_srgb(-c);
}

template <typename Fn>
ImageStatus apply_planar3(ArrayRef<f64> a, ArrayRef<f64> b, ArrayRef<f64> c, MutArrayRef<f64> o0,
                          MutArrayRef<f64> o1, MutArrayRef<f64> o2, Fn fn) noexcept {
    const usize n = a.size();
    if (b.size() < n || c.size() < n || o0.size() < n || o1.size() < n || o2.size() < n) {
        return ImageStatus::kInvalidArgument;
    }
    for (usize i = 0; i < n; ++i) {
        const Color3 out = fn(a[i], b[i], c[i]);
        o0[i] = out.c0;
        o1[i] = out.c1;
        o2[i] = out.c2;
    }
    return ImageStatus::kOk;
}

} // namespace

f64 rgb_to_gray(f64 r, f64 g, f64 b) noexcept {
    return kLumaR * r + kLumaG * g + kLumaB * b;
}

f64 srgb_to_linear(f64 c) noexcept {
    return (c <= 0.04045) ? (c / 12.92) : keel::pow((c + 0.055) / 1.055, 2.4);
}

f64 linear_to_srgb(f64 c) noexcept {
    return (c <= 0.0031308) ? (12.92 * c) : (1.055 * keel::pow(c, 1.0 / 2.4) - 0.055);
}

Color3 rgb_to_hsv(f64 r, f64 g, f64 b) noexcept {
    const f64 cmax = keel::max(r, keel::max(g, b));
    const f64 cmin = keel::min(r, keel::min(g, b));
    const f64 delta = cmax - cmin;
    f64 h = 0.0;
    if (delta > 0.0) {
        if (cmax == r) {
            h = keel::fmod((g - b) / delta, 6.0) / 6.0;
        } else if (cmax == g) {
            h = ((b - r) / delta + 2.0) / 6.0;
        } else {
            h = ((r - g) / delta + 4.0) / 6.0;
        }
        if (h < 0.0) h += 1.0;
    }
    const f64 s = (cmax == 0.0) ? 0.0 : (delta / cmax);
    return Color3{h, s, cmax};
}

Color3 hsv_to_rgb(f64 h, f64 s, f64 v) noexcept {
    h = h - keel::floor(h);
    const f64 h6 = h * 6.0;
    const i32 sector = as<i32>(keel::floor(h6));
    const f64 f = h6 - as<f64>(sector);
    const f64 p = v * (1.0 - s);
    const f64 q = v * (1.0 - s * f);
    const f64 t = v * (1.0 - s * (1.0 - f));
    switch (sector % 6) {
        case 0: return Color3{v, t, p};
        case 1: return Color3{q, v, p};
        case 2: return Color3{p, v, t};
        case 3: return Color3{p, q, v};
        case 4: return Color3{t, p, v};
        default: return Color3{v, p, q};
    }
}

Color3 rgb_to_ntsc(f64 r, f64 g, f64 b) noexcept {
    const f64 y = 0.2989360212937755 * r + 0.5870430744511212 * g + 0.1140209042551033 * b;
    const f64 i = 0.5959457430707994 * r - 0.2743886357457893 * g - 0.3215571073250100 * b;
    const f64 q = 0.2114973403068283 * r - 0.5229106903029738 * g + 0.3114133499961453 * b;
    return Color3{y, i, q};
}

Color3 ntsc_to_rgb(f64 y, f64 i, f64 q) noexcept {
    f64 r = y + 0.956 * i + 0.621 * q;
    f64 g = y - 0.272 * i - 0.647 * q;
    f64 b = y - 1.106 * i + 1.703 * q;
    if (r < 0.0) r = 0.0;
    if (g < 0.0) g = 0.0;
    if (b < 0.0) b = 0.0;
    const f64 m = keel::max(r, keel::max(g, b));
    if (m > 1.0) {
        r /= m;
        g /= m;
        b /= m;
    }
    return Color3{r, g, b};
}

Color3 rgb_to_ycbcr(f64 r, f64 g, f64 b) noexcept {
    const f64 y = (65.481 * r + 128.553 * g + 24.966 * b + 16.0) / 255.0;
    const f64 cb = (-37.797 * r - 74.203 * g + 112.0 * b + 128.0) / 255.0;
    const f64 cr = (112.0 * r - 93.786 * g - 18.214 * b + 128.0) / 255.0;
    return Color3{y, cb, cr};
}

Color3 ycbcr_to_rgb(f64 y, f64 cb, f64 cr) noexcept {
    const f64 yp = 255.0 * y - 16.0;
    const f64 cbp = 255.0 * cb - 128.0;
    const f64 crp = 255.0 * cr - 128.0;
    const f64 r = keel::clamp(
        0.0045662100456621011 * yp + 1.1808799897946412e-09 * cbp + 0.0062589289699439363 * crp,
        0.0, 1.0);
    const f64 g = keel::clamp(
        0.0045662100456621011 * yp - 0.0015363236860449021 * cbp - 0.003188110949655707 * crp,
        0.0, 1.0);
    const f64 b = keel::clamp(
        0.0045662100456621011 * yp + 0.0079107162335547414 * cbp + 1.1977497040190077e-08 * crp,
        0.0, 1.0);
    return Color3{r, g, b};
}

Color3 rgb_to_xyz(f64 r, f64 g, f64 b) noexcept {
    const f64 rl = srgb_to_linear(r);
    const f64 gl = srgb_to_linear(g);
    const f64 bl = srgb_to_linear(b);
    const f64 x = 0.4124564 * rl + 0.3575761 * gl + 0.1804375 * bl;
    const f64 y = 0.2126729 * rl + 0.7151522 * gl + 0.0721750 * bl;
    const f64 z = 0.0193339 * rl + 0.1191920 * gl + 0.9503041 * bl;
    return Color3{x, y, z};
}

Color3 xyz_to_rgb(f64 x, f64 y, f64 z) noexcept {
    const f64 rl = 3.2404542 * x - 1.5371385 * y - 0.4985314 * z;
    const f64 gl = -0.9692660 * x + 1.8760108 * y + 0.0415560 * z;
    const f64 bl = 0.0556434 * x - 0.2040259 * y + 1.0572252 * z;
    return Color3{signed_srgb_encode(rl), signed_srgb_encode(gl), signed_srgb_encode(bl)};
}

Color3 xyz_to_lab(f64 x, f64 y, f64 z) noexcept {
    const f64 fx = f_lab(x / kXyzXn);
    const f64 fy = f_lab(y / kXyzYn);
    const f64 fz = f_lab(z / kXyzZn);
    return Color3{116.0 * fy - 16.0, 500.0 * (fx - fy), 200.0 * (fy - fz)};
}

Color3 lab_to_xyz(f64 l, f64 a, f64 b) noexcept {
    const f64 fy = (l + 16.0) / 116.0;
    const f64 fx = fy + a / 500.0;
    const f64 fz = fy - b / 200.0;
    return Color3{kXyzXn * finv_lab(fx), kXyzYn * finv_lab(fy), kXyzZn * finv_lab(fz)};
}

Color3 rgb_to_lab(f64 r, f64 g, f64 b) noexcept {
    const Color3 xyz = rgb_to_xyz(r, g, b);
    return xyz_to_lab(xyz.c0, xyz.c1, xyz.c2);
}

Color3 lab_to_rgb(f64 l, f64 a, f64 b) noexcept {
    const Color3 xyz = lab_to_xyz(l, a, b);
    return xyz_to_rgb(xyz.c0, xyz.c1, xyz.c2);
}

ImageStatus rgb_to_gray_planar(ArrayRef<f64> r, ArrayRef<f64> g, ArrayRef<f64> b,
                               MutArrayRef<f64> gray) noexcept {
    const usize n = r.size();
    if (g.size() < n || b.size() < n || gray.size() < n) {
        return ImageStatus::kInvalidArgument;
    }
    for (usize i = 0; i < n; ++i) {
        gray[i] = kLumaR * r[i] + kLumaG * g[i] + kLumaB * b[i];
    }
    return ImageStatus::kOk;
}

ImageStatus rgb_to_gray_u8_planar(ArrayRef<u8> r, ArrayRef<u8> g, ArrayRef<u8> b,
                                  MutArrayRef<u8> gray) noexcept {
    const usize n = r.size();
    if (g.size() < n || b.size() < n || gray.size() < n) {
        return ImageStatus::kInvalidArgument;
    }
    for (usize i = 0; i < n; ++i) {
        const f64 y = kLumaR * as<f64>(r[i]) + kLumaG * as<f64>(g[i]) +
                      kLumaB * as<f64>(b[i]);
        const f64 clamped = keel::clamp(keel::round(y), 0.0, 255.0);
        gray[i] = as<u8>(clamped);
    }
    return ImageStatus::kOk;
}

ImageStatus rgb_to_hsv_planar(ArrayRef<f64> r, ArrayRef<f64> g, ArrayRef<f64> b, MutArrayRef<f64> h,
                              MutArrayRef<f64> s, MutArrayRef<f64> v) noexcept {
    return apply_planar3(r, g, b, h, s, v, rgb_to_hsv);
}

ImageStatus hsv_to_rgb_planar(ArrayRef<f64> h, ArrayRef<f64> s, ArrayRef<f64> v, MutArrayRef<f64> r,
                              MutArrayRef<f64> g, MutArrayRef<f64> b) noexcept {
    return apply_planar3(h, s, v, r, g, b, hsv_to_rgb);
}

ImageStatus rgb_to_ycbcr_planar(ArrayRef<f64> r, ArrayRef<f64> g, ArrayRef<f64> b,
                                MutArrayRef<f64> y, MutArrayRef<f64> cb,
                                MutArrayRef<f64> cr) noexcept {
    return apply_planar3(r, g, b, y, cb, cr, rgb_to_ycbcr);
}

ImageStatus ycbcr_to_rgb_planar(ArrayRef<f64> y, ArrayRef<f64> cb, ArrayRef<f64> cr,
                                MutArrayRef<f64> r, MutArrayRef<f64> g,
                                MutArrayRef<f64> b) noexcept {
    return apply_planar3(y, cb, cr, r, g, b, ycbcr_to_rgb);
}

ImageStatus rgb_to_xyz_planar(ArrayRef<f64> r, ArrayRef<f64> g, ArrayRef<f64> b, MutArrayRef<f64> x,
                              MutArrayRef<f64> y, MutArrayRef<f64> z) noexcept {
    return apply_planar3(r, g, b, x, y, z, rgb_to_xyz);
}

ImageStatus xyz_to_rgb_planar(ArrayRef<f64> x, ArrayRef<f64> y, ArrayRef<f64> z, MutArrayRef<f64> r,
                              MutArrayRef<f64> g, MutArrayRef<f64> b) noexcept {
    return apply_planar3(x, y, z, r, g, b, xyz_to_rgb);
}

ImageStatus xyz_to_lab_planar(ArrayRef<f64> x, ArrayRef<f64> y, ArrayRef<f64> z, MutArrayRef<f64> l,
                              MutArrayRef<f64> a, MutArrayRef<f64> b) noexcept {
    return apply_planar3(x, y, z, l, a, b, xyz_to_lab);
}

ImageStatus lab_to_xyz_planar(ArrayRef<f64> l, ArrayRef<f64> a, ArrayRef<f64> b, MutArrayRef<f64> x,
                              MutArrayRef<f64> y, MutArrayRef<f64> z) noexcept {
    return apply_planar3(l, a, b, x, y, z, lab_to_xyz);
}

ImageStatus rgb_to_lab_planar(ArrayRef<f64> r, ArrayRef<f64> g, ArrayRef<f64> b, MutArrayRef<f64> l,
                              MutArrayRef<f64> a, MutArrayRef<f64> out_b) noexcept {
    return apply_planar3(r, g, b, l, a, out_b, rgb_to_lab);
}

ImageStatus lab_to_rgb_planar(ArrayRef<f64> l, ArrayRef<f64> a, ArrayRef<f64> in_b,
                              MutArrayRef<f64> r, MutArrayRef<f64> g, MutArrayRef<f64> b) noexcept {
    return apply_planar3(l, a, in_b, r, g, b, lab_to_rgb);
}

} // namespace keel::image
