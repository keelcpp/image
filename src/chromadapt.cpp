#include "keel/image.h"

#include "keel/math.h"

namespace keel::image {

namespace {

constexpr f64 kD65[3] = {0.95047, 1.00000, 1.08883};
constexpr f64 kD50[3] = {0.96422, 1.00000, 0.82521};

constexpr f64 kMsRGBD65[9] = {
    0.4124564, 0.3575761, 0.1804375,
    0.2126729, 0.7151522, 0.0721750,
    0.0193339, 0.1191920, 0.9503041,
};

constexpr f64 kMAdobeD65[9] = {
    0.5767309, 0.1855540, 0.1881852,
    0.2973769, 0.6273491, 0.0752741,
    0.0270343, 0.0706872, 0.9911085,
};

constexpr f64 kMProPhotoD50[9] = {
    0.7976749, 0.1351917, 0.0313534,
    0.2880402, 0.7118741, 0.0000857,
    0.0000000, 0.0000000, 0.8252100,
};

constexpr f64 kMBradford[9] = {
     0.8951,  0.2664, -0.1614,
    -0.7502,  1.7135,  0.0367,
     0.0389, -0.0685,  1.0296,
};

constexpr f64 kMVonKries[9] = {
     0.4002400,  0.7076000, -0.0808100,
    -0.2263000,  1.1653200,  0.0457000,
     0.0000000,  0.0000000,  0.9182200,
};

inline void mat3_mul_vec3(const f64 m[9], const f64 v[3], f64 out[3]) noexcept {
    out[0] = m[0] * v[0] + m[1] * v[1] + m[2] * v[2];
    out[1] = m[3] * v[0] + m[4] * v[1] + m[5] * v[2];
    out[2] = m[6] * v[0] + m[7] * v[1] + m[8] * v[2];
}

inline void mat3_inv(const f64 m[9], f64 inv[9]) noexcept {
    const f64 a = m[0], b = m[1], c = m[2];
    const f64 d = m[3], e = m[4], f = m[5];
    const f64 g = m[6], h = m[7], i = m[8];
    const f64 ca = e * i - f * h;
    const f64 cb = -(d * i - f * g);
    const f64 cc = d * h - e * g;
    const f64 inv_det = 1.0 / (a * ca + b * cb + c * cc);
    inv[0] = ca * inv_det;
    inv[1] = -(b * i - c * h) * inv_det;
    inv[2] = (b * f - c * e) * inv_det;
    inv[3] = cb * inv_det;
    inv[4] = (a * i - c * g) * inv_det;
    inv[5] = -(a * f - c * d) * inv_det;
    inv[6] = cc * inv_det;
    inv[7] = -(a * h - b * g) * inv_det;
    inv[8] = (a * e - b * d) * inv_det;
}

inline void mat3_mul(const f64 a[9], const f64 b[9], f64 out[9]) noexcept {
    for (i32 i = 0; i < 3; ++i) {
        for (i32 j = 0; j < 3; ++j) {
            f64 s = 0.0;
            for (i32 k = 0; k < 3; ++k) s += a[3 * i + k] * b[3 * k + j];
            out[3 * i + j] = s;
        }
    }
}

inline f64 gamma_decode(f64 v, RgbColorSpace cs) noexcept {
    switch (cs) {
        case RgbColorSpace::kSrgb:
            if (v <= 0.04045) return v / 12.92;
            return keel::pow((v + 0.055) / 1.055, 2.4);
        case RgbColorSpace::kAdobeRgb1998:
            return keel::pow(v < 0.0 ? 0.0 : v, 2.19921875);
        case RgbColorSpace::kProPhotoRgb:
            if (v < 16.0 / 512.0) return v / 16.0;
            return keel::pow(v, 1.8);
        case RgbColorSpace::kLinearRgb:
            return v;
    }
    return v;
}

inline f64 gamma_encode(f64 v, RgbColorSpace cs) noexcept {
    switch (cs) {
        case RgbColorSpace::kSrgb:
            if (v <= 0.0031308) return v * 12.92;
            return 1.055 * keel::pow(v < 0.0 ? 0.0 : v, 1.0 / 2.4) - 0.055;
        case RgbColorSpace::kAdobeRgb1998:
            return keel::pow(v < 0.0 ? 0.0 : v, 1.0 / 2.19921875);
        case RgbColorSpace::kProPhotoRgb:
            if (v < 1.0 / 512.0) return v * 16.0;
            return keel::pow(v < 0.0 ? 0.0 : v, 1.0 / 1.8);
        case RgbColorSpace::kLinearRgb:
            return v;
    }
    return v;
}

inline const f64* cs_to_xyz_matrix(RgbColorSpace cs) noexcept {
    switch (cs) {
        case RgbColorSpace::kSrgb:
        case RgbColorSpace::kLinearRgb: return kMsRGBD65;
        case RgbColorSpace::kAdobeRgb1998: return kMAdobeD65;
        case RgbColorSpace::kProPhotoRgb: return kMProPhotoD50;
    }
    return kMsRGBD65;
}

void make_adapt(const f64 src_xyz[3], const f64 dst_xyz[3], const f64 m[9], f64 out[9]) noexcept {
    f64 lms_src[3]{};
    f64 lms_dst[3]{};
    mat3_mul_vec3(m, src_xyz, lms_src);
    mat3_mul_vec3(m, dst_xyz, lms_dst);
    for (i32 k = 0; k < 3; ++k) {
        if (keel::abs(lms_src[k]) < Limits<f64>::kEpsilon) {
            lms_src[k] = (lms_src[k] < 0.0) ? -Limits<f64>::kEpsilon : Limits<f64>::kEpsilon;
        }
    }
    const f64 d[9] = {
        lms_dst[0] / lms_src[0], 0.0, 0.0,
        0.0, lms_dst[1] / lms_src[1], 0.0,
        0.0, 0.0, lms_dst[2] / lms_src[2],
    };
    f64 m_inv[9]{};
    f64 tmp[9]{};
    mat3_inv(m, m_inv);
    mat3_mul(d, m, tmp);
    mat3_mul(m_inv, tmp, out);
}

void rgb_to_xyz_d65(const f64 rgb[3], RgbColorSpace cs, f64 xyz[3]) noexcept {
    const f64 lin[3] = {
        gamma_decode(rgb[0], cs),
        gamma_decode(rgb[1], cs),
        gamma_decode(rgb[2], cs),
    };
    f64 xyz_native[3]{};
    mat3_mul_vec3(cs_to_xyz_matrix(cs), lin, xyz_native);
    if (cs == RgbColorSpace::kProPhotoRgb) {
        f64 m_adapt[9]{};
        make_adapt(kD50, kD65, kMBradford, m_adapt);
        mat3_mul_vec3(m_adapt, xyz_native, xyz);
    } else {
        xyz[0] = xyz_native[0];
        xyz[1] = xyz_native[1];
        xyz[2] = xyz_native[2];
    }
}

void xyz_d65_to_rgb(const f64 xyz_d65[3], RgbColorSpace cs, f64 rgb[3]) noexcept {
    f64 xyz_native[3]{};
    if (cs == RgbColorSpace::kProPhotoRgb) {
        f64 m_adapt[9]{};
        make_adapt(kD65, kD50, kMBradford, m_adapt);
        mat3_mul_vec3(m_adapt, xyz_d65, xyz_native);
    } else {
        xyz_native[0] = xyz_d65[0];
        xyz_native[1] = xyz_d65[1];
        xyz_native[2] = xyz_d65[2];
    }
    f64 m_inv[9]{};
    mat3_inv(cs_to_xyz_matrix(cs), m_inv);
    f64 lin[3]{};
    mat3_mul_vec3(m_inv, xyz_native, lin);
    rgb[0] = gamma_encode(lin[0], cs);
    rgb[1] = gamma_encode(lin[1], cs);
    rgb[2] = gamma_encode(lin[2], cs);
}

} // namespace

ImageStatus chromadapt_planar(ArrayRef<f64> in_r, ArrayRef<f64> in_g, ArrayRef<f64> in_b,
                              Color3 illuminant_rgb01, ChromAdaptMethod method,
                              RgbColorSpace color_space, MutArrayRef<f64> out_r,
                              MutArrayRef<f64> out_g, MutArrayRef<f64> out_b) noexcept {
    const usize n = in_r.size();
    if (n == 0u || in_g.size() != n || in_b.size() != n || out_r.size() < n || out_g.size() < n ||
        out_b.size() < n) {
        return ImageStatus::kInvalidArgument;
    }
    if (!keel::is_finite(illuminant_rgb01.c0) || !keel::is_finite(illuminant_rgb01.c1) ||
        !keel::is_finite(illuminant_rgb01.c2) || illuminant_rgb01.c0 < 0.0 ||
        illuminant_rgb01.c1 < 0.0 || illuminant_rgb01.c2 < 0.0) {
        return ImageStatus::kInvalidArgument;
    }
    if (illuminant_rgb01.c0 == 0.0 && illuminant_rgb01.c1 == 0.0 && illuminant_rgb01.c2 == 0.0) {
        return ImageStatus::kInvalidArgument;
    }

    const f64 ill_rgb[3] = {illuminant_rgb01.c0, illuminant_rgb01.c1, illuminant_rgb01.c2};
    f64 ill_xyz[3]{};
    rgb_to_xyz_d65(ill_rgb, color_space, ill_xyz);
    for (i32 k = 0; k < 3; ++k) {
        if (ill_xyz[k] == 0.0) ill_xyz[k] = Limits<f64>::kEpsilon;
    }
    const f64 y_val = ill_xyz[1];
    ill_xyz[0] /= y_val;
    ill_xyz[1] /= y_val;
    ill_xyz[2] /= y_val;

    if (method == ChromAdaptMethod::kSimple) {
        f64 ill_norm[3]{};
        xyz_d65_to_rgb(ill_xyz, color_space, ill_norm);
        for (i32 k = 0; k < 3; ++k) {
            ill_norm[k] = keel::abs(ill_norm[k]);
            if (ill_norm[k] == 0.0) ill_norm[k] = Limits<f64>::kEpsilon;
        }
        for (usize i = 0; i < n; ++i) {
            out_r[i] = in_r[i] / ill_norm[0];
            out_g[i] = in_g[i] / ill_norm[1];
            out_b[i] = in_b[i] / ill_norm[2];
        }
        return ImageStatus::kOk;
    }

    // Precompute combined linear-RGB -> linear-RGB 3x3 matrix for speed while staying exact.
    f64 m_adapt[9]{};
    const f64* lms_mat = (method == ChromAdaptMethod::kBradford) ? kMBradford : kMVonKries;
    make_adapt(ill_xyz, kD65, lms_mat, m_adapt);

    f64 m_to_d65[9]{};
    f64 m_from_d65[9]{};
    const f64* m_cs = cs_to_xyz_matrix(color_space);
    f64 m_cs_inv[9]{};
    mat3_inv(m_cs, m_cs_inv);

    if (color_space == RgbColorSpace::kProPhotoRgb) {
        f64 m_50_65[9]{};
        f64 m_65_50[9]{};
        make_adapt(kD50, kD65, kMBradford, m_50_65);
        make_adapt(kD65, kD50, kMBradford, m_65_50);
        mat3_mul(m_50_65, m_cs, m_to_d65);
        mat3_mul(m_cs_inv, m_65_50, m_from_d65);
    } else {
        for (i32 i = 0; i < 9; ++i) {
            m_to_d65[i] = m_cs[i];
            m_from_d65[i] = m_cs_inv[i];
        }
    }

    for (usize i = 0; i < n; ++i) {
        const f64 lin_in[3] = {
            gamma_decode(in_r[i], color_space),
            gamma_decode(in_g[i], color_space),
            gamma_decode(in_b[i], color_space),
        };
        f64 xyz[3]{};
        f64 xyz_ad[3]{};
        f64 lin_out[3]{};
        mat3_mul_vec3(m_to_d65, lin_in, xyz);
        mat3_mul_vec3(m_adapt, xyz, xyz_ad);
        mat3_mul_vec3(m_from_d65, xyz_ad, lin_out);
        out_r[i] = gamma_encode(lin_out[0], color_space);
        out_g[i] = gamma_encode(lin_out[1], color_space);
        out_b[i] = gamma_encode(lin_out[2], color_space);
    }
    return ImageStatus::kOk;
}

} // namespace keel::image
