#include "keel/image.h"

#include "keel/algo.h"
#include "keel/hash_map.h"
#include "keel/hash_set.h"
#include "keel/math.h"
#include "keel/static_array.h"

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

namespace {

constexpr f64 kMBt2020D65[9] = {
    0.637010191411101,     0.144615027396969, 0.168844781191930,
    0.262721717361640,     0.677989275502262, 0.059289007136098,
    4.994515405547190e-17, 0.028072328847647, 1.060757671152350,
};

struct BtWideParams {
    f64 alpha;
    f64 beta;
    f64 black_level;
    f64 nominal_peak;
};

bool resolve_bt_params(i32 bps, WideColorSpace cs, BtWideParams& out) noexcept {
    if (bps != 10 && bps != 12) {
        return false;
    }
    if (cs == WideColorSpace::kBt2020) {
        if (bps == 10) {
            out = BtWideParams{1.099, 0.018, 64.0, 940.0};
        } else {
            out = BtWideParams{1.0993, 0.0181, 256.0, 3760.0};
        }
        return true;
    }
    out = (bps == 10) ? BtWideParams{1.099, 0.018, 64.0, 940.0}
                      : BtWideParams{1.099, 0.018, 256.0, 3760.0};
    return true;
}

f64 bt_rgb2lin(f64 v, f64 alpha, f64 beta) noexcept {
    if (v < 4.5 * beta) {
        return v / 4.5;
    }
    return pow((v + alpha - 1.0) / alpha, 1.0 / 0.45);
}

f64 bt_lin2rgb(f64 v, f64 alpha, f64 beta) noexcept {
    if (v < beta) {
        return 4.5 * v;
    }
    return alpha * pow(v, 0.45) - (alpha - 1.0);
}

f64 hlg_rgb2lin(f64 v) noexcept {
    constexpr f64 a = 0.17883277;
    constexpr f64 b = 1.0 - 4.0 * a;
    const f64 c = 0.5 - a * log(4.0 * a);
    if (v <= 0.5) {
        return (v * v) / 3.0;
    }
    return (exp((v - c) / a) + b) / 12.0;
}

f64 hlg_lin2rgb(f64 v) noexcept {
    constexpr f64 a = 0.17883277;
    constexpr f64 b = 1.0 - 4.0 * a;
    const f64 c = 0.5 - a * log(4.0 * a);
    if (v <= 1.0 / 12.0) {
        return sqrt(3.0 * v);
    }
    return a * log(12.0 * v - b) + c;
}

void mat3_inv(const f64 m[9], f64 inv[9]) noexcept {
    const f64 a = m[0];
    const f64 b = m[1];
    const f64 c = m[2];
    const f64 d = m[3];
    const f64 e = m[4];
    const f64 f = m[5];
    const f64 g = m[6];
    const f64 h = m[7];
    const f64 i = m[8];
    const f64 c00 = e * i - f * h;
    const f64 c01 = -(d * i - f * g);
    const f64 c02 = d * h - e * g;
    const f64 det = a * c00 + b * c01 + c * c02;
    const f64 id = 1.0 / det;
    inv[0] = c00 * id;
    inv[1] = -(b * i - c * h) * id;
    inv[2] = (b * f - c * e) * id;
    inv[3] = c01 * id;
    inv[4] = (a * i - c * g) * id;
    inv[5] = -(a * f - c * d) * id;
    inv[6] = c02 * id;
    inv[7] = -(a * h - b * g) * id;
    inv[8] = (a * e - b * d) * id;
}

u16 clamp_round_half_up_u16(f64 v) noexcept {
    if (v < 0.0) return 0;
    if (v > 65535.0) return 65535;
    return as<u16>(v + 0.5);
}

u16 clamp_round_u16(f64 v) noexcept {
    const f64 r = round(v);
    if (r < 0.0) return 0;
    if (r > 65535.0) return 65535;
    return as<u16>(r);
}

} // namespace

ImageStatus rgbwide_to_ycbcr_planar(ArrayRef<u16> r, ArrayRef<u16> g, ArrayRef<u16> b,
                                    i32 bits_per_sample, MutArrayRef<u16> y, MutArrayRef<u16> cb,
                                    MutArrayRef<u16> cr) noexcept {
    if (bits_per_sample != 10 && bits_per_sample != 12) {
        return ImageStatus::kInvalidArgument;
    }
    const usize n = r.size();
    if (g.size() != n || b.size() != n || y.size() != n || cb.size() != n || cr.size() != n) {
        return ImageStatus::kInvalidArgument;
    }
    const f64 black_level = (bits_per_sample == 10) ? 64.0 : 256.0;
    const f64 nominal_peak = (bits_per_sample == 10) ? 940.0 : 3760.0;
    const f64 nominal_range = nominal_peak - black_level;
    const f64 scale = (bits_per_sample == 10) ? 4.0 : 16.0;
    for (usize i = 0; i < n; ++i) {
        const f64 rn = (as<f64>(r[i]) - black_level) / nominal_range;
        const f64 gn = (as<f64>(g[i]) - black_level) / nominal_range;
        const f64 bn = (as<f64>(b[i]) - black_level) / nominal_range;
        const f64 yv = 0.2627 * rn + 0.6780 * gn + 0.0593 * bn;
        const f64 cbv = (bn - yv) / 1.8814;
        const f64 crv = (rn - yv) / 1.4746;
        y[i] = clamp_round_half_up_u16((219.0 * yv + 16.0) * scale);
        cb[i] = clamp_round_half_up_u16((224.0 * cbv + 128.0) * scale);
        cr[i] = clamp_round_half_up_u16((224.0 * crv + 128.0) * scale);
    }
    return ImageStatus::kOk;
}

ImageStatus ycbcr_to_rgbwide_planar(ArrayRef<u16> y, ArrayRef<u16> cb, ArrayRef<u16> cr,
                                    i32 bits_per_sample, MutArrayRef<u16> r, MutArrayRef<u16> g,
                                    MutArrayRef<u16> b) noexcept {
    if (bits_per_sample != 10 && bits_per_sample != 12) {
        return ImageStatus::kInvalidArgument;
    }
    const usize n = y.size();
    if (cb.size() != n || cr.size() != n || r.size() != n || g.size() != n || b.size() != n) {
        return ImageStatus::kInvalidArgument;
    }
    const f64 yzero = (bits_per_sample == 10) ? 64.0 : 256.0;
    const f64 yrange = (bits_per_sample == 10) ? 876.0 : 3504.0;
    const f64 chromazero = (bits_per_sample == 10) ? 512.0 : 2048.0;
    const f64 chromarange = (bits_per_sample == 10) ? 896.0 : 3584.0;
    const f64 black_level = yzero;
    const f64 nominal_range = yrange;
    for (usize i = 0; i < n; ++i) {
        const f64 yn = (as<f64>(y[i]) - yzero) / yrange;
        const f64 cbn = (as<f64>(cb[i]) - chromazero) / chromarange;
        const f64 crn = (as<f64>(cr[i]) - chromazero) / chromarange;
        const f64 rv = 1.4746 * crn + yn;
        const f64 bv = 1.8814 * cbn + yn;
        const f64 gv = (yn - 0.2627 * rv - 0.0593 * bv) / 0.6780;
        r[i] = clamp_round_half_up_u16(rv * nominal_range + black_level);
        g[i] = clamp_round_half_up_u16(gv * nominal_range + black_level);
        b[i] = clamp_round_half_up_u16(bv * nominal_range + black_level);
    }
    return ImageStatus::kOk;
}

ImageStatus rgbwide_to_xyz_planar(ArrayRef<f64> r, ArrayRef<f64> g, ArrayRef<f64> b,
                                  i32 bits_per_sample, WideColorSpace cs, MutArrayRef<f64> x,
                                  MutArrayRef<f64> y, MutArrayRef<f64> z) noexcept {
    BtWideParams p{};
    if (!resolve_bt_params(bits_per_sample, cs, p)) {
        return ImageStatus::kInvalidArgument;
    }
    const usize n = r.size();
    if (g.size() != n || b.size() != n || x.size() != n || y.size() != n || z.size() != n) {
        return ImageStatus::kInvalidArgument;
    }
    const f64 nominal_range = p.nominal_peak - p.black_level;
    const bool use_hlg = (cs == WideColorSpace::kBt2100Hlg);
    for (usize i = 0; i < n; ++i) {
        const f64 nr = (r[i] - p.black_level) / nominal_range;
        const f64 ng = (g[i] - p.black_level) / nominal_range;
        const f64 nb = (b[i] - p.black_level) / nominal_range;
        const f64 lr = use_hlg ? hlg_rgb2lin(nr) : bt_rgb2lin(nr, p.alpha, p.beta);
        const f64 lg = use_hlg ? hlg_rgb2lin(ng) : bt_rgb2lin(ng, p.alpha, p.beta);
        const f64 lb = use_hlg ? hlg_rgb2lin(nb) : bt_rgb2lin(nb, p.alpha, p.beta);
        x[i] = kMBt2020D65[0] * lr + kMBt2020D65[1] * lg + kMBt2020D65[2] * lb;
        y[i] = kMBt2020D65[3] * lr + kMBt2020D65[4] * lg + kMBt2020D65[5] * lb;
        z[i] = kMBt2020D65[6] * lr + kMBt2020D65[7] * lg + kMBt2020D65[8] * lb;
    }
    return ImageStatus::kOk;
}

ImageStatus xyz_to_rgbwide_planar(ArrayRef<f64> x, ArrayRef<f64> y, ArrayRef<f64> z,
                                  i32 bits_per_sample, WideColorSpace cs, MutArrayRef<u16> r,
                                  MutArrayRef<u16> g, MutArrayRef<u16> b) noexcept {
    BtWideParams p{};
    if (!resolve_bt_params(bits_per_sample, cs, p)) {
        return ImageStatus::kInvalidArgument;
    }
    const usize n = x.size();
    if (y.size() != n || z.size() != n || r.size() != n || g.size() != n || b.size() != n) {
        return ImageStatus::kInvalidArgument;
    }
    const f64 nominal_range = p.nominal_peak - p.black_level;
    const bool use_hlg = (cs == WideColorSpace::kBt2100Hlg);
    f64 minv[9];
    mat3_inv(kMBt2020D65, minv);
    for (usize i = 0; i < n; ++i) {
        const f64 lr = max(0.0, minv[0] * x[i] + minv[1] * y[i] + minv[2] * z[i]);
        const f64 lg = max(0.0, minv[3] * x[i] + minv[4] * y[i] + minv[5] * z[i]);
        const f64 lb = max(0.0, minv[6] * x[i] + minv[7] * y[i] + minv[8] * z[i]);
        const f64 nr = use_hlg ? hlg_lin2rgb(lr) : bt_lin2rgb(lr, p.alpha, p.beta);
        const f64 ng = use_hlg ? hlg_lin2rgb(lg) : bt_lin2rgb(lg, p.alpha, p.beta);
        const f64 nb = use_hlg ? hlg_lin2rgb(lb) : bt_lin2rgb(lb, p.alpha, p.beta);
        r[i] = clamp_round_u16(nr * nominal_range + p.black_level);
        g[i] = clamp_round_u16(ng * nominal_range + p.black_level);
        b[i] = clamp_round_u16(nb * nominal_range + p.black_level);
    }
    return ImageStatus::kOk;
}

ImageStatus rgb_to_ind_nearest_planar(ArrayRef<f64> r, ArrayRef<f64> g, ArrayRef<f64> b,
                                      ArrayRef<f64> cmap_r, ArrayRef<f64> cmap_g,
                                      ArrayRef<f64> cmap_b,
                                      MutArrayRef<u32> out_idx_0based) noexcept {
    const usize n = r.size();
    const usize k_cnt = cmap_r.size();
    if (g.size() != n || b.size() != n || out_idx_0based.size() != n || cmap_g.size() != k_cnt
        || cmap_b.size() != k_cnt || k_cnt == 0) {
        return ImageStatus::kInvalidArgument;
    }
    for (usize i = 0; i < n; ++i) {
        const f64 rv = r[i];
        const f64 gv = g[i];
        const f64 bv = b[i];
        f64 best_dist = Limits<f64>::infinity();
        usize best_k = 0;
        for (usize k = 0; k < k_cnt; ++k) {
            const f64 dr = rv - cmap_r[k];
            const f64 dg = gv - cmap_g[k];
            const f64 db = bv - cmap_b[k];
            const f64 d2 = dr * dr + dg * dg + db * db;
            if (d2 < best_dist) {
                best_dist = d2;
                best_k = k;
            }
        }
        out_idx_0based[i] = as<u32>(best_k);
    }
    return ImageStatus::kOk;
}

ImageStatus colormap_unique(ArrayRef<usize> idx_0based_in, ArrayRef<f64> map_r, ArrayRef<f64> map_g,
                            ArrayRef<f64> map_b, MutArrayRef<usize> out_idx_0based,
                            Array<f64>& out_r, Array<f64>& out_g, Array<f64>& out_b,
                            Allocator* alloc) {
    const usize m = map_r.size();
    const usize n = idx_0based_in.size();
    if (map_g.size() != m || map_b.size() != m || out_idx_0based.size() != n) {
        return ImageStatus::kInvalidArgument;
    }
    out_r.clear();
    out_g.clear();
    out_b.clear();
    if (m == 0) {
        return ImageStatus::kOk;
    }

    Allocator* a = alloc ? alloc : default_allocator();
    constexpr f64 kTol = 1.0 / 1024.0;
    Array<f64> qr(a);
    Array<f64> qg(a);
    Array<f64> qb(a);
    qr.resize(m);
    qg.resize(m);
    qb.resize(m);
    for (usize k = 0; k < m; ++k) {
        qr[k] = round(map_r[k] / kTol) * kTol;
        qg[k] = round(map_g[k] / kTol) * kTol;
        qb[k] = round(map_b[k] / kTol) * kTol;
    }

    Array<usize> ndx(a);
    ndx.resize(m);
    for (usize k = 0; k < m; ++k) {
        ndx[k] = k;
    }
    sort(ndx.begin(), ndx.end(), [&](usize x, usize y) noexcept {
        if (qb[x] != qb[y]) return qb[x] < qb[y];
        if (qg[x] != qg[y]) return qg[x] < qg[y];
        return qr[x] < qr[y];
    });

    Array<usize> pos(a);
    pos.resize(m);
    for (usize k = 0; k < m; ++k) {
        pos[ndx[k]] = k;
    }

    Array<u8> dup(a);
    dup.assign(m > 0 ? (m - 1) : 0, u8{0});
    for (usize k = 0; k + 1 < m; ++k) {
        const usize x = ndx[k];
        const usize y = ndx[k + 1];
        const bool same = (abs(qr[x] - qr[y]) < kTol) && (abs(qg[x] - qg[y]) < kTol)
                       && (abs(qb[x] - qb[y]) < kTol);
        dup[k] = same ? u8{1} : u8{0};
    }

    Array<usize> loc(a);
    loc.resize(m);
    usize csum = 0;
    for (usize k = 0; k < m; ++k) {
        loc[k] = k - csum;
        if (k < dup.size()) {
            csum += dup[k];
        }
    }

    for (usize i = 0; i < n; ++i) {
        const usize orig = idx_0based_in[i];
        if (orig >= m) {
            return ImageStatus::kInvalidArgument;
        }
        out_idx_0based[i] = loc[pos[orig]];
    }

    Array<usize> kept_ndx(a);
    kept_ndx.reserve(m);
    for (usize k = 0; k < m; ++k) {
        if (!(k < dup.size() && dup[k])) {
            kept_ndx.push_back(ndx[k]);
        }
    }
    const usize nmap1 = kept_ndx.size();

    Array<usize> nuse(a);
    nuse.assign(nmap1, usize{0});
    for (usize i = 0; i < n; ++i) {
        if (out_idx_0based[i] < nmap1) {
            ++nuse[out_idx_0based[i]];
        }
    }

    Array<usize> loc2(a);
    loc2.resize(nmap1);
    usize unused_so_far = 0;
    for (usize k = 0; k < nmap1; ++k) {
        if (nuse[k] == 0) {
            ++unused_so_far;
        }
        loc2[k] = k - unused_so_far;
    }
    for (usize i = 0; i < n; ++i) {
        out_idx_0based[i] = loc2[out_idx_0based[i]];
    }

    for (usize k = 0; k < nmap1; ++k) {
        if (nuse[k] > 0) {
            const usize src = kept_ndx[k];
            out_r.push_back(qr[src]);
            out_g.push_back(qg[src]);
            out_b.push_back(qb[src]);
        }
    }
    return ImageStatus::kOk;
}

ImageStatus greedy_graph_coloring_2d(ArrayRef<i32> labels_col_major, usize height, usize width,
                                     ArrayRef<f64> cmap_r, ArrayRef<f64> cmap_g,
                                     ArrayRef<f64> cmap_b, Array<f64>& out_r, Array<f64>& out_g,
                                     Array<f64>& out_b, Allocator* alloc) {
    const usize total = height * width;
    const usize nc_in = cmap_r.size();
    if (labels_col_major.size() != total || cmap_g.size() != nc_in || cmap_b.size() != nc_in
        || nc_in == 0) {
        return ImageStatus::kInvalidArgument;
    }
    Allocator* a = alloc ? alloc : default_allocator();

    bool tf_zero = false;
    for (usize i = 0; i < total; ++i) {
        if (labels_col_major[i] == 0) {
            tf_zero = true;
            break;
        }
    }

    Array<i32> lab(a);
    lab.resize(total);
    for (usize i = 0; i < total; ++i) {
        lab[i] = labels_col_major[i] + (tf_zero ? 1 : 0);
    }

    Array<f64> cr(a);
    Array<f64> cg(a);
    Array<f64> cb(a);
    if (tf_zero) {
        cr.reserve(nc_in + 1);
        cg.reserve(nc_in + 1);
        cb.reserve(nc_in + 1);
        cr.push_back(cmap_r[0]);
        cg.push_back(cmap_g[0]);
        cb.push_back(cmap_b[0]);
    }
    for (usize i = 0; i < nc_in; ++i) {
        cr.push_back(cmap_r[i]);
        cg.push_back(cmap_g[i]);
        cb.push_back(cmap_b[i]);
    }
    const usize nc = cr.size();

    // Build 8-conn adjacency edges between distinct labels.
    struct Edge {
        i32 u;
        i32 v;
    };
    Array<Edge> edges(a);
    i32 max_label = 0;
    for (usize r = 0; r < height; ++r) {
        for (usize c = 0; c < width; ++c) {
            const i32 u = lab[c * height + r];
            if (u > max_label) max_label = u;
            static constexpr i32 kOff[4][2] = {{1, 0}, {0, 1}, {1, 1}, {1, -1}};
            for (i32 oi = 0; oi < 4; ++oi) {
                const usize rr = r + as<usize>(kOff[oi][0]);
                const i64 cc = as<i64>(c) + kOff[oi][1];
                if (rr >= height || cc < 0 || as<usize>(cc) >= width) {
                    continue;
                }
                const i32 v = lab[as<usize>(cc) * height + rr];
                if (u == v) {
                    continue;
                }
                const i32 x = min(u, v);
                const i32 y = max(u, v);
                edges.push_back(Edge{x, y});
            }
        }
    }

    auto copy_fallback = [&]() {
        out_r.resize(nc_in);
        out_g.resize(nc_in);
        out_b.resize(nc_in);
        for (usize i = 0; i < nc_in; ++i) {
            out_r[i] = cmap_r[i];
            out_g[i] = cmap_g[i];
            out_b[i] = cmap_b[i];
        }
    };

    if (edges.empty() || max_label <= 0) {
        copy_fallback();
        return ImageStatus::kOk;
    }

    sort(edges.begin(), edges.end(), [](const Edge& x, const Edge& y) noexcept {
        if (x.u != y.u) return x.u < y.u;
        return x.v < y.v;
    });
    usize u_cnt = 1;
    for (usize i = 1; i < edges.size(); ++i) {
        if (edges[i].u != edges[u_cnt - 1].u || edges[i].v != edges[u_cnt - 1].v) {
            edges[u_cnt++] = edges[i];
        }
    }
    edges.resize(u_cnt);

    // Chromatic upper bound (max degree + 1).
    HashMap<i32, i32> deg(a);
    for (usize i = 0; i < edges.size(); ++i) {
        if (i32* p = deg.get(edges[i].u)) {
            ++(*p);
        } else {
            deg.put(edges[i].u, 1);
        }
        if (i32* p = deg.get(edges[i].v)) {
            ++(*p);
        } else {
            deg.put(edges[i].v, 1);
        }
    }
    i32 max_deg = 0;
    for (const auto& kv : deg) {
        if (kv.value > max_deg) {
            max_deg = kv.value;
        }
    }
    usize pal_k = as<usize>(max_deg + 1);
    if (pal_k > nc) {
        pal_k = nc;
    }

    // Select pal_k maximally distinct colors from (cr, cg, cb).
    Array<f64> pr(a);
    Array<f64> pg(a);
    Array<f64> pb(a);
    pr.resize(pal_k);
    pg.resize(pal_k);
    pb.resize(pal_k);
    Array<u8> picked(a);
    picked.assign(nc, u8{0});
    Array<f64> dist_sum(a);
    dist_sum.assign(nc, 0.0);
    const usize last = nc - 1;
    picked[last] = 1;
    pr[0] = cr[last];
    pg[0] = cg[last];
    pb[0] = cb[last];
    for (usize i = 0; i < nc; ++i) {
        if (!picked[i]) {
            const f64 dr = cr[i] - cr[last];
            const f64 dg = cg[i] - cg[last];
            const f64 db = cb[i] - cb[last];
            dist_sum[i] = sqrt(dr * dr + dg * dg + db * db);
        }
    }
    for (usize k = 1; k < pal_k; ++k) {
        i64 best = -1;
        f64 bv = -1.0;
        for (usize i = 0; i < nc; ++i) {
            if (!picked[i] && dist_sum[i] > bv) {
                bv = dist_sum[i];
                best = as<i64>(i);
            }
        }
        if (best < 0) {
            break;
        }
        const usize bu = as<usize>(best);
        picked[bu] = 1;
        pr[k] = cr[bu];
        pg[k] = cg[bu];
        pb[k] = cb[bu];
        for (usize i = 0; i < nc; ++i) {
            if (!picked[i]) {
                const f64 dr = cr[i] - cr[bu];
                const f64 dg = cg[i] - cg[bu];
                const f64 db = cb[i] - cb[bu];
                dist_sum[i] += sqrt(dr * dr + dg * dg + db * db);
            }
        }
    }

    // Build CSR adjacency list for nodes 0..max_label.
    const usize num_nodes = as<usize>(max_label);
    Array<usize> adj_cnt(a);
    adj_cnt.assign(num_nodes + 1, usize{0});
    for (usize i = 0; i < edges.size(); ++i) {
        if (edges[i].u >= 0 && edges[i].v >= 0) {
            ++adj_cnt[as<usize>(edges[i].u)];
            ++adj_cnt[as<usize>(edges[i].v)];
        }
    }
    Array<usize> adj_off(a);
    adj_off.resize(num_nodes + 2);
    adj_off[0] = 0;
    for (usize i = 0; i <= num_nodes; ++i) {
        adj_off[i + 1] = adj_off[i] + adj_cnt[i];
    }
    Array<i32> adj_data(a);
    adj_data.resize(adj_off[num_nodes + 1]);
    Array<usize> adj_cur(a);
    adj_cur.resize(num_nodes + 1);
    for (usize i = 0; i <= num_nodes; ++i) {
        adj_cur[i] = adj_off[i];
    }
    for (usize i = 0; i < edges.size(); ++i) {
        if (edges[i].u >= 0 && edges[i].v >= 0) {
            adj_data[adj_cur[as<usize>(edges[i].u)]++] = edges[i].v;
            adj_data[adj_cur[as<usize>(edges[i].v)]++] = edges[i].u;
        }
    }
    for (usize i = 0; i <= num_nodes; ++i) {
        sort(adj_data.data() + adj_off[i], adj_data.data() + adj_off[i + 1]);
    }

    i32 first_node = Limits<i32>::kMax;
    for (usize i = 0; i < total; ++i) {
        if (lab[i] > 0 && lab[i] < first_node) {
            first_node = lab[i];
        }
    }
    if (first_node == Limits<i32>::kMax) {
        copy_fallback();
        return ImageStatus::kOk;
    }

    Array<i32> colored_idx(a);
    colored_idx.assign(num_nodes + 1, i32{0});
    Array<u8> visited(a);
    visited.assign(num_nodes + 1, u8{0});
    Array<i32> queue(a);
    queue.reserve(num_nodes);
    queue.push_back(first_node);
    visited[as<usize>(first_node)] = 1;

    Array<f64> full_r(a);
    Array<f64> full_g(a);
    Array<f64> full_b(a);
    full_r.assign(nc, 0.0);
    full_g.assign(nc, 0.0);
    full_b.assign(nc, 0.0);

    usize head = 0;
    Array<i32> vis_nb(a);
    while (head < queue.size()) {
        const i32 cur = queue[head++];
        vis_nb.clear();
        const usize beg = adj_off[as<usize>(cur)];
        const usize end = adj_off[as<usize>(cur) + 1];
        for (usize j = beg; j < end; ++j) {
            const i32 nb = adj_data[j];
            if (!visited[as<usize>(nb)]) {
                visited[as<usize>(nb)] = 1;
                queue.push_back(nb);
            } else {
                vis_nb.push_back(nb);
            }
        }
        i32 new_idx = 1;
        if (head > 1) {
            HashSet<i32> used(a);
            for (usize j = 0; j < vis_nb.size(); ++j) {
                const i32 ci = colored_idx[as<usize>(vis_nb[j])];
                if (ci != 0) {
                    used.insert(ci);
                }
            }
            while (used.contains(new_idx)) {
                ++new_idx;
            }
            if (new_idx > as<i32>(pal_k)) {
                new_idx = as<i32>(pal_k);
            }
        }
        colored_idx[as<usize>(cur)] = new_idx;
        if (cur >= 1 && as<usize>(cur - 1) < nc) {
            full_r[as<usize>(cur - 1)] = pr[as<usize>(new_idx - 1)];
            full_g[as<usize>(cur - 1)] = pg[as<usize>(new_idx - 1)];
            full_b[as<usize>(cur - 1)] = pb[as<usize>(new_idx - 1)];
        }
    }

    for (usize i = 0; i < nc; ++i) {
        const usize l1 = i + 1;
        if (l1 <= num_nodes && !visited[l1]) {
            full_r[i] = cr[i];
            full_g[i] = cg[i];
            full_b[i] = cb[i];
        }
    }

    const usize offset = tf_zero ? 1 : 0;
    out_r.resize(nc_in);
    out_g.resize(nc_in);
    out_b.resize(nc_in);
    for (usize i = 0; i < nc_in; ++i) {
        out_r[i] = full_r[i + offset];
        out_g[i] = full_g[i + offset];
        out_b[i] = full_b[i + offset];
    }
    return ImageStatus::kOk;
}

} // namespace keel::image
