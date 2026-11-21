#pragma once

#include "keel/array.h"
#include "keel/complex.h"
#include "keel/core.h"
#include "keel/memory.h"
#include "keel/ref.h"

namespace keel::image {

enum class ImageStatus : u8 {
    kOk = 0,
    kInvalidArgument,
};

struct Color3 {
    f64 c0 = 0.0;
    f64 c1 = 0.0;
    f64 c2 = 0.0;
};

struct OtsuResult {
    f64 threshold = 0.0;     // Normalized threshold in [0, 1] (`bin_level / (L - 1)`)
    f64 effectiveness = 0.0; // Separability metric eta = sigma_b^2 / sigma_T^2 in [0, 1]
    f64 bin_level = 0.0;     // 0-based mean of tied optimal bin indices
};

// ============================================================================
// Color-Space Conversions (Per-Pixel)
// ============================================================================

[[nodiscard]] f64 rgb_to_gray(f64 r, f64 g, f64 b) noexcept;

[[nodiscard]] f64 srgb_to_linear(f64 c) noexcept;
[[nodiscard]] f64 linear_to_srgb(f64 c) noexcept;

[[nodiscard]] Color3 rgb_to_hsv(f64 r, f64 g, f64 b) noexcept;
[[nodiscard]] Color3 hsv_to_rgb(f64 h, f64 s, f64 v) noexcept;

[[nodiscard]] Color3 rgb_to_ntsc(f64 r, f64 g, f64 b) noexcept;
[[nodiscard]] Color3 ntsc_to_rgb(f64 y, f64 i, f64 q) noexcept;

[[nodiscard]] Color3 rgb_to_ycbcr(f64 r, f64 g, f64 b) noexcept;
[[nodiscard]] Color3 ycbcr_to_rgb(f64 y, f64 cb, f64 cr) noexcept;

[[nodiscard]] Color3 rgb_to_xyz(f64 r, f64 g, f64 b) noexcept;
[[nodiscard]] Color3 xyz_to_rgb(f64 x, f64 y, f64 z) noexcept;

[[nodiscard]] Color3 xyz_to_lab(f64 x, f64 y, f64 z) noexcept;
[[nodiscard]] Color3 lab_to_xyz(f64 l, f64 a, f64 b) noexcept;

[[nodiscard]] Color3 rgb_to_lab(f64 r, f64 g, f64 b) noexcept;
[[nodiscard]] Color3 lab_to_rgb(f64 l, f64 a, f64 b) noexcept;

// ============================================================================
// Color-Space Conversions (Planar Batches)
// ============================================================================
// Planar layout: each channel is a contiguous ArrayRef of length `npix`.

[[nodiscard]] ImageStatus rgb_to_gray_planar(ArrayRef<f64> r, ArrayRef<f64> g, ArrayRef<f64> b,
                                             MutArrayRef<f64> gray) noexcept;
[[nodiscard]] ImageStatus rgb_to_gray_u8_planar(ArrayRef<u8> r, ArrayRef<u8> g, ArrayRef<u8> b,
                                                MutArrayRef<u8> gray) noexcept;

[[nodiscard]] ImageStatus rgb_to_hsv_planar(ArrayRef<f64> r, ArrayRef<f64> g, ArrayRef<f64> b,
                                            MutArrayRef<f64> h, MutArrayRef<f64> s,
                                            MutArrayRef<f64> v) noexcept;
[[nodiscard]] ImageStatus hsv_to_rgb_planar(ArrayRef<f64> h, ArrayRef<f64> s, ArrayRef<f64> v,
                                            MutArrayRef<f64> r, MutArrayRef<f64> g,
                                            MutArrayRef<f64> b) noexcept;

[[nodiscard]] ImageStatus rgb_to_ycbcr_planar(ArrayRef<f64> r, ArrayRef<f64> g, ArrayRef<f64> b,
                                              MutArrayRef<f64> y, MutArrayRef<f64> cb,
                                              MutArrayRef<f64> cr) noexcept;
[[nodiscard]] ImageStatus ycbcr_to_rgb_planar(ArrayRef<f64> y, ArrayRef<f64> cb, ArrayRef<f64> cr,
                                              MutArrayRef<f64> r, MutArrayRef<f64> g,
                                              MutArrayRef<f64> b) noexcept;

[[nodiscard]] ImageStatus rgb_to_xyz_planar(ArrayRef<f64> r, ArrayRef<f64> g, ArrayRef<f64> b,
                                            MutArrayRef<f64> x, MutArrayRef<f64> y,
                                            MutArrayRef<f64> z) noexcept;
[[nodiscard]] ImageStatus xyz_to_rgb_planar(ArrayRef<f64> x, ArrayRef<f64> y, ArrayRef<f64> z,
                                            MutArrayRef<f64> r, MutArrayRef<f64> g,
                                            MutArrayRef<f64> b) noexcept;

[[nodiscard]] ImageStatus xyz_to_lab_planar(ArrayRef<f64> x, ArrayRef<f64> y, ArrayRef<f64> z,
                                            MutArrayRef<f64> l, MutArrayRef<f64> a,
                                            MutArrayRef<f64> b) noexcept;
[[nodiscard]] ImageStatus lab_to_xyz_planar(ArrayRef<f64> l, ArrayRef<f64> a, ArrayRef<f64> b,
                                            MutArrayRef<f64> x, MutArrayRef<f64> y,
                                            MutArrayRef<f64> z) noexcept;

[[nodiscard]] ImageStatus rgb_to_lab_planar(ArrayRef<f64> r, ArrayRef<f64> g, ArrayRef<f64> b,
                                            MutArrayRef<f64> l, MutArrayRef<f64> a,
                                            MutArrayRef<f64> out_b) noexcept;
[[nodiscard]] ImageStatus lab_to_rgb_planar(ArrayRef<f64> l, ArrayRef<f64> a, ArrayRef<f64> in_b,
                                            MutArrayRef<f64> r, MutArrayRef<f64> g,
                                            MutArrayRef<f64> b) noexcept;

// ============================================================================
// Image Quality, Statistics & 2D Correlation
// ============================================================================

[[nodiscard]] f64 mse(ArrayRef<f64> a, ArrayRef<f64> b) noexcept;
[[nodiscard]] f64 mse_c64(ArrayRef<c64> a, ArrayRef<c64> b) noexcept;
[[nodiscard]] f64 psnr(ArrayRef<f64> a, ArrayRef<f64> b, f64 peak = 1.0) noexcept;

[[nodiscard]] f64 mean2(ArrayRef<f64> img) noexcept;
[[nodiscard]] f64 std2(ArrayRef<f64> img) noexcept;

// Normalized 2D Gaussian kernel (column-major `height x width`).
[[nodiscard]] ImageStatus gaussian_kernel_2d(usize height, usize width, f64 sigma,
                                             MutArrayRef<f64> out_col_major) noexcept;

// Same-size 2D correlation with replicate border padding (column-major).
[[nodiscard]] ImageStatus correlate_2d_replicate(ArrayRef<f64> img_col_major, usize height,
                                                 usize width, ArrayRef<f64> kernel_col_major,
                                                 usize kh, usize kw,
                                                 MutArrayRef<f64> out_col_major) noexcept;

// Structural Similarity Index (Wang et al., 11x11 Gaussian sigma=1.5, replicate padding).
// Expects column-major `height x width` images scaled to `[0, dynamic_range]`.
[[nodiscard]] f64 ssim(ArrayRef<f64> a_col_major, ArrayRef<f64> b_col_major, usize height,
                       usize width, f64 dynamic_range = 1.0, Allocator* alloc = nullptr);

// ============================================================================
// Histograms & Thresholding
// ============================================================================

// Uniform histogram of `img` over `[min_val, max_val]` into `counts` (`counts.size()` bins).
[[nodiscard]] ImageStatus histogram(ArrayRef<f64> img, f64 min_val, f64 max_val,
                                    MutArrayRef<f64> counts) noexcept;

// Single-level Otsu threshold on histogram `hist_counts`.
[[nodiscard]] OtsuResult otsu_threshold(ArrayRef<f64> hist_counts, Allocator* alloc = nullptr);

// ============================================================================
// Bayer Demosaicing (Malvar-He-Cutler 5x5)
// ============================================================================

enum class BayerPattern : u8 {
    kRggb = 0,
    kBggr,
    kGrbg,
    kGbrg,
};

// Demosaic an even-sized `height x width` column-major Bayer mosaic into planar R/G/B,
// clamping outputs to `[0, max_val]` (pass `max_val <= 0` to disable upper clamping).
[[nodiscard]] ImageStatus demosaic_mhc(ArrayRef<f64> mosaic_col_major, usize height, usize width,
                                       BayerPattern pattern, f64 max_val, MutArrayRef<f64> out_r,
                                       MutArrayRef<f64> out_g, MutArrayRef<f64> out_b) noexcept;

// ============================================================================
// Texture Analysis (GLCM)
// ============================================================================

struct GlcmProps {
    f64 contrast = 0.0;
    f64 correlation = 0.0;
    f64 energy = 0.0;
    f64 homogeneity = 0.0;
};

// Gray-level co-occurrence matrix (`num_levels x num_levels` column-major).
[[nodiscard]] ImageStatus graycomatrix(ArrayRef<f64> img_col_major, usize height, usize width,
                                       i32 num_levels, i32 off_r, i32 off_c, f64 g_low, f64 g_high,
                                       bool symmetric, MutArrayRef<f64> out_glcm_col_major,
                                       Allocator* alloc = nullptr);

// Haralick summary statistics on a square `num_levels x num_levels` GLCM.
[[nodiscard]] ImageStatus graycoprops(ArrayRef<f64> glcm_col_major, usize num_levels,
                                      GlcmProps& out) noexcept;

// ============================================================================
// Chromatic Adaptation (Bradford / Von Kries / Simple)
// ============================================================================

enum class ChromAdaptMethod : u8 {
    kBradford = 0,
    kVonKries,
    kSimple,
};

enum class RgbColorSpace : u8 {
    kSrgb = 0,
    kAdobeRgb1998,
    kProPhotoRgb,
    kLinearRgb,
};

// White-balance chromatic adaptation on planar [0, 1] encoded RGB channels.
[[nodiscard]] ImageStatus chromadapt_planar(ArrayRef<f64> in_r, ArrayRef<f64> in_g,
                                            ArrayRef<f64> in_b, Color3 illuminant_rgb01,
                                            ChromAdaptMethod method, RgbColorSpace color_space,
                                            MutArrayRef<f64> out_r, MutArrayRef<f64> out_g,
                                            MutArrayRef<f64> out_b) noexcept;

// ============================================================================
// 3D Filter Kernels & 2D Windowed FIR Design
// ============================================================================

enum class Axis3D : u8 {
    kX = 0,
    kY,
    kZ,
};

[[nodiscard]] ImageStatus fspecial3_average(usize h, usize w, usize p,
                                            MutArrayRef<f64> out_col_major) noexcept;
[[nodiscard]] ImageStatus fspecial3_gaussian(usize h, usize w, usize p, f64 sr, f64 sc, f64 sp,
                                             MutArrayRef<f64> out_col_major) noexcept;
[[nodiscard]] ImageStatus fspecial3_ellipsoid(f64 ar, f64 ac, f64 ap, Array<f64>& out_col_major,
                                              usize& out_h, usize& out_w, usize& out_p,
                                              Allocator* alloc = nullptr);
[[nodiscard]] ImageStatus fspecial3_laplacian(f64 gamma1, f64 gamma2,
                                              MutArrayRef<f64> out_col_major_3x3x3) noexcept;
[[nodiscard]] ImageStatus fspecial3_log(usize h, usize w, usize p, f64 sr, f64 sc, f64 sp,
                                        MutArrayRef<f64> out_col_major, Allocator* alloc = nullptr);
[[nodiscard]] ImageStatus fspecial3_gradient(bool sobel, Axis3D axis,
                                             MutArrayRef<f64> out_col_major_3x3x3) noexcept;
[[nodiscard]] ImageStatus fwind2(ArrayRef<f64> hd_col_major, ArrayRef<f64> win_col_major,
                                 usize height, usize width,
                                 MutArrayRef<f64> out_col_major) noexcept;

// ============================================================================
// Multiscale Frangi Vesselness (fibermetric) & Local Laplacian Filtering
// ============================================================================

[[nodiscard]] ImageStatus fibermetric_2d(ArrayRef<f64> img_col_major, usize height, usize width,
                                         ArrayRef<f64> thickness, f64 sensitivity, bool bright,
                                         MutArrayRef<f64> out_col_major,
                                         Allocator* alloc = nullptr);
[[nodiscard]] ImageStatus fibermetric_3d(ArrayRef<f64> vol_col_major, usize height, usize width,
                                         usize depth, ArrayRef<f64> thickness, f64 sensitivity,
                                         bool bright, MutArrayRef<f64> out_col_major,
                                         Allocator* alloc = nullptr);

// Fast Local Laplacian filter (Aubry-Paris-Hasinoff-Kautz-Durand 2014).
// Pass `num_intensity_levels <= 0` for automatic level selection from `alpha`.
[[nodiscard]] ImageStatus local_laplacian_gray(ArrayRef<f32> in_col_major, usize height,
                                               usize width, f32 sigma, f32 alpha, f32 beta,
                                               i32 num_intensity_levels,
                                               MutArrayRef<f32> out_col_major,
                                               Allocator* alloc = nullptr);
[[nodiscard]] ImageStatus local_laplacian_rgb(ArrayRef<f32> in_r, ArrayRef<f32> in_g,
                                              ArrayRef<f32> in_b, usize height, usize width,
                                              f32 sigma, f32 alpha, f32 beta,
                                              i32 num_intensity_levels, bool process_luminance,
                                              MutArrayRef<f32> out_r, MutArrayRef<f32> out_g,
                                              MutArrayRef<f32> out_b, Allocator* alloc = nullptr);

} // namespace keel::image
