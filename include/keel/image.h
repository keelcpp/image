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

// Wide-gamut BT.2020 / BT.2100 (10- or 12-bit narrow-range uint16) conversions.
enum class WideColorSpace : u8 {
    kBt2020 = 0,
    kBt2100Pq,
    kBt2100Hlg,
};

[[nodiscard]] ImageStatus rgbwide_to_ycbcr_planar(ArrayRef<u16> r, ArrayRef<u16> g, ArrayRef<u16> b,
                                                  i32 bits_per_sample, MutArrayRef<u16> y,
                                                  MutArrayRef<u16> cb,
                                                  MutArrayRef<u16> cr) noexcept;
[[nodiscard]] ImageStatus ycbcr_to_rgbwide_planar(ArrayRef<u16> y, ArrayRef<u16> cb,
                                                  ArrayRef<u16> cr, i32 bits_per_sample,
                                                  MutArrayRef<u16> r, MutArrayRef<u16> g,
                                                  MutArrayRef<u16> b) noexcept;
[[nodiscard]] ImageStatus rgbwide_to_xyz_planar(ArrayRef<f64> r, ArrayRef<f64> g, ArrayRef<f64> b,
                                                i32 bits_per_sample, WideColorSpace cs,
                                                MutArrayRef<f64> x, MutArrayRef<f64> y,
                                                MutArrayRef<f64> z) noexcept;
[[nodiscard]] ImageStatus xyz_to_rgbwide_planar(ArrayRef<f64> x, ArrayRef<f64> y, ArrayRef<f64> z,
                                                i32 bits_per_sample, WideColorSpace cs,
                                                MutArrayRef<u16> r, MutArrayRef<u16> g,
                                                MutArrayRef<u16> b) noexcept;

// Palette nearest-neighbor RGB quantization, colormap deduplication, and graph coloring.
[[nodiscard]] ImageStatus rgb_to_ind_nearest_planar(ArrayRef<f64> r, ArrayRef<f64> g,
                                                    ArrayRef<f64> b, ArrayRef<f64> cmap_r,
                                                    ArrayRef<f64> cmap_g, ArrayRef<f64> cmap_b,
                                                    MutArrayRef<u32> out_idx_0based) noexcept;
[[nodiscard]] ImageStatus colormap_unique(ArrayRef<usize> idx_0based_in, ArrayRef<f64> map_r,
                                          ArrayRef<f64> map_g, ArrayRef<f64> map_b,
                                          MutArrayRef<usize> out_idx_0based, Array<f64>& out_r,
                                          Array<f64>& out_g, Array<f64>& out_b,
                                          Allocator* alloc = nullptr);
[[nodiscard]] ImageStatus greedy_graph_coloring_2d(ArrayRef<i32> labels_col_major, usize height,
                                                   usize width, ArrayRef<f64> cmap_r,
                                                   ArrayRef<f64> cmap_g, ArrayRef<f64> cmap_b,
                                                   Array<f64>& out_r, Array<f64>& out_g,
                                                   Array<f64>& out_b, Allocator* alloc = nullptr);

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

// ============================================================================
// Illumination Estimation & Perceptual Color Difference (CIE94 / CIEDE2000)
// ============================================================================

enum class ColorDiffStandard : u8 {
    kCie94 = 0,
    kCiede2000,
};

// White-patch top-percentile illuminant estimator (`mask` empty => all pixels selected).
[[nodiscard]] ImageStatus illumwhite_planar(ArrayRef<f64> r, ArrayRef<f64> g, ArrayRef<f64> b,
                                            ArrayRef<u8> mask, f64 percentile, Color3& out,
                                            Allocator* alloc = nullptr);

// Grey-world trimmed Minkowski-norm illuminant estimator (`mask` empty => all pixels selected).
[[nodiscard]] ImageStatus illumgray_planar(ArrayRef<f64> r, ArrayRef<f64> g, ArrayRef<f64> b,
                                           ArrayRef<u8> mask, f64 p_lo, f64 p_hi, f64 norm_exp,
                                           Color3& out, Allocator* alloc = nullptr);

// Cheng-Prasad-Brown (2014) PCA illuminant estimator (`mask` empty => all pixels selected).
[[nodiscard]] ImageStatus illumpca_planar(ArrayRef<f64> r, ArrayRef<f64> g, ArrayRef<f64> b,
                                          ArrayRef<u8> mask, f64 percentage, Color3& out,
                                          Allocator* alloc = nullptr);

[[nodiscard]] f64 delta_e_94(Color3 lab1, Color3 lab2, f64 kL = 1.0, f64 kC = 1.0, f64 kH = 1.0,
                             f64 K1 = 0.045, f64 K2 = 0.015) noexcept;

[[nodiscard]] f64 delta_e_2000(Color3 lab1, Color3 lab2, f64 kL = 1.0, f64 kC = 1.0, f64 kH = 1.0,
                               f64 K1 = 0.045, f64 K2 = 0.015) noexcept;

[[nodiscard]] ImageStatus colordiff_lab_planar(ArrayRef<f64> l1, ArrayRef<f64> a1, ArrayRef<f64> b1,
                                               ArrayRef<f64> l2, ArrayRef<f64> a2, ArrayRef<f64> b2,
                                               ColorDiffStandard standard, f64 kL, f64 kC, f64 kH,
                                               f64 K1, f64 K2, MutArrayRef<f64> out) noexcept;

// ============================================================================
// Geometric Resampling, Rotation, Translation & Burt-Adelson Pyramids
// ============================================================================

enum class ResizeKernel : u8 {
    kNearest = 0,
    kBox,
    kTriangle,
    kCubic,
    kLanczos2,
    kLanczos3,
};

// Column-major `in_h x in_w x channels` -> `out_h x out_w x channels`.
[[nodiscard]] ImageStatus resize_2d(ArrayRef<f64> in_col_major, usize in_h, usize in_w,
                                    usize channels, usize out_h, usize out_w, f64 scale_y,
                                    f64 scale_x, ResizeKernel kernel, bool antialias,
                                    MutArrayRef<f64> out_col_major, Allocator* alloc = nullptr);

// Column-major `in_r x in_c x in_d` -> `out_r x out_c x out_d`.
[[nodiscard]] ImageStatus resize_3d(ArrayRef<f64> in_col_major, usize in_r, usize in_c, usize in_d,
                                    usize out_r, usize out_c, usize out_d, f64 scale_y, f64 scale_x,
                                    f64 scale_z, ResizeKernel kernel, bool antialias,
                                    MutArrayRef<f64> out_col_major, Allocator* alloc = nullptr);

enum class RotateInterp : u8 {
    kNearest = 0,
    kBilinear,
    kBicubic,
};

void rotate_2d_dims(usize in_h, usize in_w, f64 angle_deg, bool crop, usize& out_h,
                    usize& out_w) noexcept;

[[nodiscard]] ImageStatus rotate_2d(ArrayRef<f64> in_col_major, usize in_h, usize in_w,
                                    usize channels, f64 angle_deg, RotateInterp interp, bool crop,
                                    MutArrayRef<f64> out_col_major, usize out_h,
                                    usize out_w) noexcept;

[[nodiscard]] ImageStatus rotate_2d(ArrayRef<f64> in_col_major, usize in_h, usize in_w,
                                    usize channels, f64 angle_deg, bool nearest, bool crop,
                                    MutArrayRef<f64> out_col_major, usize out_h,
                                    usize out_w) noexcept;

[[nodiscard]] ImageStatus translate_2d(ArrayRef<f64> in_col_major, usize h, usize w, usize channels,
                                       f64 dx, f64 dy, MutArrayRef<f64> out_col_major) noexcept;

[[nodiscard]] ImageStatus pyramid_reduce_2d(ArrayRef<f64> in_col_major, usize h, usize w,
                                            usize channels, MutArrayRef<f64> out_col_major,
                                            Allocator* alloc = nullptr);

[[nodiscard]] ImageStatus pyramid_expand_2d(ArrayRef<f64> in_col_major, usize h, usize w,
                                            usize channels, MutArrayRef<f64> out_col_major,
                                            Allocator* alloc = nullptr);

// ============================================================================
// 2D FIR Filter Design (fsamp2 / ftrans2 / fwind1 / fwind2)
// ============================================================================

[[nodiscard]] ImageStatus fsamp2_real(ArrayRef<f64> hd_col_major, usize h, usize w,
                                      MutArrayRef<f64> out_col_major, Allocator* alloc = nullptr);

[[nodiscard]] ImageStatus fsamp2_complex(ArrayRef<c64> hd_col_major, usize h, usize w,
                                         MutArrayRef<c64> out_col_major,
                                         Allocator* alloc = nullptr);

// Pass empty `t_col_major` (mt = nt = 0) for the default 3x3 McClellan transformation.
[[nodiscard]] ImageStatus ftrans2(ArrayRef<f64> b, ArrayRef<f64> t_col_major, usize mt, usize nt,
                                  Array<f64>& out_col_major, usize& out_h, usize& out_w,
                                  Allocator* alloc = nullptr);

// Pass empty `win2` for Huang's radial 1-window method; non-empty `win2` for separable `win2 * win1^T`.
[[nodiscard]] ImageStatus fwind1_real(ArrayRef<f64> hd_col_major, usize hh, usize wh,
                                      ArrayRef<f64> win1, ArrayRef<f64> win2,
                                      MutArrayRef<f64> out_col_major, Allocator* alloc = nullptr);

[[nodiscard]] ImageStatus fwind1_complex(ArrayRef<c64> hd_col_major, usize hh, usize wh,
                                         ArrayRef<c64> win1, ArrayRef<c64> win2,
                                         MutArrayRef<c64> out_col_major,
                                         Allocator* alloc = nullptr);

[[nodiscard]] ImageStatus fwind2_real(ArrayRef<f64> hd_col_major, usize hh, usize wh,
                                      ArrayRef<f64> win_col_major, usize hw, usize ww,
                                      MutArrayRef<f64> out_col_major, Allocator* alloc = nullptr);

[[nodiscard]] ImageStatus fwind2_complex(ArrayRef<c64> hd_col_major, usize hh, usize wh,
                                         ArrayRef<c64> win_col_major, usize hw, usize ww,
                                         MutArrayRef<c64> out_col_major,
                                         Allocator* alloc = nullptr);

// ============================================================================
// Box Filter, Guided Image Filter & Dark-Channel Prior Dehazing
// ============================================================================

[[nodiscard]] ImageStatus box_filter_2d_replicate(ArrayRef<f64> in_col_major, usize height,
                                                  usize width, i32 filter_size,
                                                  MutArrayRef<f64> out_col_major,
                                                  Allocator* alloc = nullptr);

[[nodiscard]] ImageStatus guided_filter_gray(ArrayRef<f64> img_col_major,
                                             ArrayRef<f64> guide_col_major, usize height,
                                             usize width, i32 nhood, f64 eps,
                                             MutArrayRef<f64> out_col_major,
                                             Allocator* alloc = nullptr);

enum class DehazeMethod : u8 {
    kSimpleDcp = 0,
    kApproxDcp,
};

enum class DehazeContrast : u8 {
    kGlobal = 0,
    kBoost,
    kNone,
};

// Dark-channel prior dehazing on planar [0, 1] `height x width x channels` (channels = 1 or 3).
// Pass empty `atm_override` to estimate atmospheric light automatically.
// `out_thickness_col_major` has length `height * width` (may be empty if unused).
// `out_atm` has length `channels` (may be empty if unused).
[[nodiscard]] ImageStatus dehaze(ArrayRef<f32> in_planar_col_major, usize height, usize width,
                                 usize channels, f64 amount, DehazeMethod method,
                                 ArrayRef<f32> atm_override, DehazeContrast contrast,
                                 f64 boost_amount, MutArrayRef<f32> out_planar_col_major,
                                 MutArrayRef<f64> out_thickness_col_major, MutArrayRef<f64> out_atm,
                                 Allocator* alloc = nullptr);

// ============================================================================
// Contrast Enhancement, Adaptive Thresholding, Multi-Otsu & Flat-Fielding
// ============================================================================

// Separable 2D Gaussian blur (`symmetric_pad == false` => replicate, `true` => half-sample mirror).
[[nodiscard]] ImageStatus gaussian_filter_2d(ArrayRef<f64> in_col_major, usize height, usize width,
                                             f64 sigma, i32 filter_size, bool symmetric_pad,
                                             MutArrayRef<f64> out_col_major,
                                             Allocator* alloc = nullptr);

// Per-channel histogram stretch limits on unit [0, 1] planar data (`out_lo_hi` has length 2*channels).
[[nodiscard]] ImageStatus stretchlim_unit(ArrayRef<f64> unit_planar, usize plane_len,
                                          usize channels, i32 nbins, f64 low_tol, f64 high_tol,
                                          MutArrayRef<f64> out_lo_hi, Allocator* alloc = nullptr);

// Gamma-mapped contrast adjustment on unit [0, 1] data.
[[nodiscard]] ImageStatus imadjust_unit(ArrayRef<f64> unit_in, f64 low_in, f64 high_in,
                                        f64 low_out, f64 high_out, f64 gamma,
                                        MutArrayRef<f64> unit_out) noexcept;

// Global histogram equalization on unit [0, 1] data.
[[nodiscard]] ImageStatus histeq_unit(ArrayRef<f64> unit_in, i32 nbins, MutArrayRef<f64> unit_out,
                                      Allocator* alloc = nullptr);

enum class ClaheDistribution : u8 {
    kUniform = 0,
    kRayleigh,
    kExponential,
};

// Contrast Limited Adaptive Histogram Equalization (Zuiderveld 1994) on 2D column-major image.
[[nodiscard]] ImageStatus clahe_2d(ArrayRef<f64> img_col_major, usize height, usize width,
                                   i32 num_tiles_r, i32 num_tiles_c, f64 clip_limit, i32 nbins,
                                   ClaheDistribution dist, f64 alpha, f64 class_lo, f64 class_hi,
                                   bool range_original, MutArrayRef<f64> out_col_major,
                                   Allocator* alloc = nullptr);

// Multi-level Otsu thresholding (1 <= num_levels <= 5; exact for 1-2, global DP for 3-5).
[[nodiscard]] ImageStatus multithresh_otsu(ArrayRef<f64> data, i32 num_levels, bool integer_scale,
                                           MutArrayRef<f64> out_thresh, f64& out_effectiveness,
                                           Allocator* alloc = nullptr);

// Gonzalez histogram matching of `in_unit` to `ref_unit` (`out_hgram` may be empty or length `nbins`).
[[nodiscard]] ImageStatus imhistmatch_unit(ArrayRef<f64> in_unit, ArrayRef<f64> ref_unit, i32 npts,
                                           i32 nbins, MutArrayRef<f64> out_unit,
                                           MutArrayRef<f64> out_hgram, Allocator* alloc = nullptr);

// Locally adaptive threshold map on [0, 1] 2D image (`gaussian == false` => box mean).
[[nodiscard]] ImageStatus adaptthresh_2d(ArrayRef<f64> unit_col_major, usize height, usize width,
                                         f64 sensitivity, i32 neighborhood, bool gaussian,
                                         MutArrayRef<f64> out_col_major,
                                         Allocator* alloc = nullptr);

// 2D flat-field illumination correction (`mask_col_major` may be empty or length `height * width`).
[[nodiscard]] ImageStatus flatfield_2d(ArrayRef<f64> unit_planar_col_major, usize height,
                                       usize width, usize pages, f64 sigma,
                                       ArrayRef<u8> mask_col_major,
                                       MutArrayRef<f64> out_planar_col_major,
                                       Allocator* alloc = nullptr);

enum class WcodematMode : u8 {
    kMat = 0,
    kRow,
    kCol,
};

// Pseudo-color quantization into 1..nb levels (`mat` / `row` / `col`).
[[nodiscard]] ImageStatus wcodemat(ArrayRef<f64> x_col_major, usize height, usize width,
                                   usize pages, i32 nb, WcodematMode mode, bool absol,
                                   MutArrayRef<f64> out_col_major) noexcept;

// Shannon entropy in bits (`-sum p_i log2(p_i)`) from histogram bin counts.
[[nodiscard]] f64 shannon_entropy(ArrayRef<f64> hist_counts) noexcept;

// ============================================================================
// Morphological Operations (2D & 3D)
// ============================================================================

struct StrelMask {
    usize rows = 0;
    usize cols = 0;
    Array<u8> mask_col_major; // 0/1 column-major (rows * cols)
};

[[nodiscard]] ImageStatus strel_square(i32 n, StrelMask& out, Allocator* alloc = nullptr);
[[nodiscard]] ImageStatus strel_rect(i32 rows, i32 cols, StrelMask& out,
                                     Allocator* alloc = nullptr);
[[nodiscard]] ImageStatus strel_diamond(i32 radius, StrelMask& out, Allocator* alloc = nullptr);
[[nodiscard]] ImageStatus strel_disk(f64 radius, i32 n_periodic, StrelMask& out,
                                     Allocator* alloc = nullptr);
[[nodiscard]] ImageStatus strel_line(f64 length, f64 theta_deg, StrelMask& out,
                                     Allocator* alloc = nullptr);

// Flat structuring element erosion/dilation on column-major u8 (logical or uint8) images.
// Uses van Herk-Gil-Werman O(width) fast path for column-contiguous centered SEs.
[[nodiscard]] ImageStatus morph_erode_u8(ArrayRef<u8> src_col_major, usize height, usize width,
                                         ArrayRef<u8> se_col_major, usize se_h, usize se_w,
                                         MutArrayRef<u8> dst_col_major, Allocator* alloc = nullptr);
[[nodiscard]] ImageStatus morph_dilate_u8(ArrayRef<u8> src_col_major, usize height, usize width,
                                          ArrayRef<u8> se_col_major, usize se_h, usize se_w,
                                          MutArrayRef<u8> dst_col_major,
                                          Allocator* alloc = nullptr);

// Flat structuring element erosion/dilation on column-major f64 images (preserves +-Inf).
[[nodiscard]] ImageStatus morph_erode_f64(ArrayRef<f64> src_col_major, usize height, usize width,
                                          ArrayRef<u8> se_col_major, usize se_h, usize se_w,
                                          MutArrayRef<f64> dst_col_major) noexcept;
[[nodiscard]] ImageStatus morph_dilate_f64(ArrayRef<f64> src_col_major, usize height, usize width,
                                           ArrayRef<u8> se_col_major, usize se_h, usize se_w,
                                           MutArrayRef<f64> dst_col_major) noexcept;

// Morphological reconstruction by dilation (Vincent 1993 fast hybrid algorithm).
[[nodiscard]] ImageStatus imreconstruct_f64(ArrayRef<f64> marker_col_major,
                                            ArrayRef<f64> mask_col_major, usize height, usize width,
                                            i32 conn, MutArrayRef<f64> out_col_major,
                                            Allocator* alloc = nullptr);

// Fill interior holes in binary (0/1) or grayscale f64 2D images.
[[nodiscard]] ImageStatus imfill_holes_binary(ArrayRef<u8> bw_col_major, usize height, usize width,
                                              i32 conn, MutArrayRef<u8> out_col_major,
                                              Allocator* alloc = nullptr);
[[nodiscard]] ImageStatus imfill_holes_gray(ArrayRef<f64> img_col_major, usize height, usize width,
                                            i32 conn, MutArrayRef<f64> out_col_major,
                                            Allocator* alloc = nullptr);

// Regional maxima / minima (output 0/1 mask).
[[nodiscard]] ImageStatus imregionalmax_2d(ArrayRef<f64> img_col_major, usize height, usize width,
                                           i32 conn, MutArrayRef<u8> out_col_major,
                                           Allocator* alloc = nullptr);
[[nodiscard]] ImageStatus imregionalmin_2d(ArrayRef<f64> img_col_major, usize height, usize width,
                                           i32 conn, MutArrayRef<u8> out_col_major,
                                           Allocator* alloc = nullptr);

// H-maxima / H-minima transforms.
[[nodiscard]] ImageStatus imhmax_2d(ArrayRef<f64> img_col_major, usize height, usize width, f64 h,
                                    i32 conn, MutArrayRef<f64> out_col_major,
                                    Allocator* alloc = nullptr);
[[nodiscard]] ImageStatus imhmin_2d(ArrayRef<f64> img_col_major, usize height, usize width, f64 h,
                                    i32 conn, MutArrayRef<f64> out_col_major,
                                    Allocator* alloc = nullptr);

// Minima imposition (Soille 1999). `is_float_class` chooses h = (max-min)/1000 vs h = 1.
[[nodiscard]] ImageStatus imimposemin_2d(ArrayRef<f64> img_col_major, ArrayRef<u8> bw_col_major,
                                         usize height, usize width, bool is_float_class, i32 conn,
                                         MutArrayRef<f64> out_col_major,
                                         Allocator* alloc = nullptr);

// Clear or keep foreground components touching the 2D image border.
[[nodiscard]] ImageStatus imclearborder_binary(ArrayRef<u8> bw_col_major, usize height, usize width,
                                               i32 conn, MutArrayRef<u8> out_col_major,
                                               Allocator* alloc = nullptr);
[[nodiscard]] ImageStatus imkeepborder_binary(ArrayRef<u8> bw_col_major, usize height, usize width,
                                              i32 conn, MutArrayRef<u8> out_col_major,
                                              Allocator* alloc = nullptr);

// Pack / unpack binary 2D images into 32-bit column words (`packed_h = ceil(height / 32)`).
[[nodiscard]] ImageStatus bwpack_2d(ArrayRef<u8> bw_col_major, usize height, usize width,
                                    MutArrayRef<u32> out_col_major) noexcept;
[[nodiscard]] ImageStatus bwunpack_2d(ArrayRef<u32> packed_col_major, usize packed_h, usize width,
                                      usize out_h, MutArrayRef<u8> out_col_major) noexcept;

// General n x n binary neighborhood LUT filter (`lut.size() == 2^(n*n)`).
[[nodiscard]] ImageStatus bwlookup_nxn(ArrayRef<u8> bw_col_major, usize height, usize width,
                                       ArrayRef<f64> lut, usize n,
                                       MutArrayRef<f64> out_col_major) noexcept;

enum class BwmorphOp : u8 {
    kDilate = 0,
    kErode,
    kBridge,
    kClean,
    kDiag,
    kEndpoints,
    kFatten,
    kFill,
    kHbreak,
    kMajority,
    kPerim4,
    kPerim8,
    kRemove,
    kOpen,
    kClose,
    kBothat,
    kTophat,
    kThin,
    kSkel,
    kShrink,
    kSpur,
    kThicken,
    kBranchpoints,
};

// Binary 2D morphology (`iterations < 0` repeats until convergence).
[[nodiscard]] ImageStatus bwmorph_2d(ArrayRef<u8> bw_col_major, usize height, usize width,
                                     BwmorphOp op, i32 iterations, MutArrayRef<u8> out_col_major,
                                     Allocator* alloc = nullptr);

enum class Bwmorph3Op : u8 {
    kBranchpoints = 0,
    kClean,
    kEndpoints,
    kFill,
    kMajority,
    kRemove,
};

// Binary 3D volume morphology (column-major height x width x pages).
[[nodiscard]] ImageStatus bwmorph3_3d(ArrayRef<u8> vol_col_major, usize height, usize width,
                                      usize pages, Bwmorph3Op op,
                                      MutArrayRef<u8> out_col_major) noexcept;

struct BoundaryPoint {
    i64 row = 0; // 0-based
    i64 col = 0; // 0-based
};

// Moore-Neighbor boundary tracing on 2D binary image.
// `first_step_dir`: 0..7 for 8-conn (N,NE,E,SE,S,SW,W,NW); 0..3 for 4-conn (N,E,S,W).
// `max_points == 0` traces the full closed contour.
[[nodiscard]] ImageStatus bwtraceboundary_2d(ArrayRef<u8> bw_col_major, usize height, usize width,
                                             i64 start_row, i64 start_col, i32 first_step_dir,
                                             i32 conn, usize max_points, bool clockwise,
                                             Array<BoundaryPoint>& out_points);

// ============================================================================
// Spatial, Statistical, 3D & Denoising Filters
// ============================================================================

enum class PadMode : u8 {
    kConstant = 0,
    kReplicate,
    kSymmetric,
    kCircular,
};

struct Kernel2D {
    usize rows = 0;
    usize cols = 0;
    Array<f64> data_col_major;
};

// Pad a 2D/3D column-major array (`height x width x pages`).
[[nodiscard]] ImageStatus padarray_f64(ArrayRef<f64> src_col_major, usize height, usize width,
                                       usize pages, i32 pre_r, i32 post_r, i32 pre_c, i32 post_c,
                                       i32 pre_p, i32 post_p, PadMode mode, f64 pad_value,
                                       MutArrayRef<f64> out_col_major) noexcept;

// Standard 2D filter kernel factories (`fspecial`).
[[nodiscard]] ImageStatus fspecial_average(i32 rows, i32 cols, Kernel2D& out,
                                           Allocator* alloc = nullptr);
[[nodiscard]] ImageStatus fspecial_gaussian(i32 rows, i32 cols, f64 sigma, Kernel2D& out,
                                            Allocator* alloc = nullptr);
[[nodiscard]] ImageStatus fspecial_laplacian(f64 alpha, Kernel2D& out, Allocator* alloc = nullptr);
[[nodiscard]] ImageStatus fspecial_log(i32 rows, i32 cols, f64 sigma, Kernel2D& out,
                                       Allocator* alloc = nullptr);
[[nodiscard]] ImageStatus fspecial_sobel(Kernel2D& out, Allocator* alloc = nullptr);
[[nodiscard]] ImageStatus fspecial_prewitt(Kernel2D& out, Allocator* alloc = nullptr);
[[nodiscard]] ImageStatus fspecial_disk(f64 radius, Kernel2D& out, Allocator* alloc = nullptr);
[[nodiscard]] ImageStatus fspecial_motion(f64 length, f64 theta_deg, Kernel2D& out,
                                          Allocator* alloc = nullptr);
[[nodiscard]] ImageStatus fspecial_unsharp(f64 alpha, Kernel2D& out, Allocator* alloc = nullptr);

// 2D correlation (`flip_kernel == false`) or convolution (`flip_kernel == true`).
// Output size is `(height + k_h - 1) x (width + k_w - 1)` when `full == true`, else `height x width`.
[[nodiscard]] ImageStatus imfilter_2d(ArrayRef<f64> img_col_major, usize height, usize width,
                                      ArrayRef<f64> kernel_col_major, usize k_h, usize k_w,
                                      PadMode boundary, f64 pad_value, bool full, bool flip_kernel,
                                      MutArrayRef<f64> out_col_major) noexcept;

// Replicate-padded 2D box filter on uint8 using integer summed-area table fast path.
[[nodiscard]] ImageStatus box_filter_2d_u8_replicate(ArrayRef<u8> in_col_major, usize height,
                                                     usize width, i32 filter_size,
                                                     MutArrayRef<u8> out_col_major,
                                                     Allocator* alloc = nullptr);

// 2D / 3D box filter on precomputed integral images `(h1 x w1 [x pages])` or `(h1 x w1 x d1)`.
[[nodiscard]] ImageStatus integral_box_filter_2d(ArrayRef<f64> sat_col_major, usize h1, usize w1,
                                                 usize pages, i32 fh, i32 fw, f64 norm_factor,
                                                 MutArrayRef<f64> out_col_major) noexcept;
[[nodiscard]] ImageStatus integral_box_filter_3d(ArrayRef<f64> sat_col_major, usize h1, usize w1,
                                                 usize d1, i32 fh, i32 fw, i32 fp, f64 norm_factor,
                                                 MutArrayRef<f64> out_col_major) noexcept;

// 2D / 3D mode filter (`fd == 0` => 2D; `pad_mode` in {kSymmetric, kReplicate, kConstant}).
[[nodiscard]] ImageStatus modefilt_u8(ArrayRef<u8> src_col_major, usize height, usize width,
                                      usize pages, i32 fh, i32 fw, i32 fd, PadMode pad_mode,
                                      MutArrayRef<u8> out_col_major) noexcept;
[[nodiscard]] ImageStatus modefilt_f64(ArrayRef<f64> src_col_major, usize height, usize width,
                                       usize pages, i32 fh, i32 fw, i32 fd, PadMode pad_mode,
                                       MutArrayRef<f64> out_col_major, Allocator* alloc = nullptr);

// 2D median filter (zero-padded borders; Huang O(W) sliding histogram for odd u8, med9 for 3x3 f64).
[[nodiscard]] ImageStatus medfilt2_u8(ArrayRef<u8> src_col_major, usize height, usize width,
                                      i32 rows, i32 cols, MutArrayRef<u8> out_col_major,
                                      Allocator* alloc = nullptr);
[[nodiscard]] ImageStatus medfilt2_f64(ArrayRef<f64> src_col_major, usize height, usize width,
                                       i32 rows, i32 cols, MutArrayRef<f64> out_col_major,
                                       Allocator* alloc = nullptr);

// Unsharp masking (`I + amount * mask * (I - gauss(I, radius))`).
[[nodiscard]] ImageStatus imsharpen_2d(ArrayRef<f64> img_col_major, usize height, usize width,
                                       f64 radius, f64 amount, f64 threshold,
                                       MutArrayRef<f64> out_col_major, Allocator* alloc = nullptr);

// Edge-preserving bilateral filter (replicate border).
[[nodiscard]] ImageStatus imbilatfilt_2d(ArrayRef<f64> img_col_major, usize height, usize width,
                                         f64 degree_of_smoothing, f64 spatial_sigma,
                                         MutArrayRef<f64> out_col_major,
                                         Allocator* alloc = nullptr);

// Local standard deviation, general order-statistic, local entropy, and local range filters.
[[nodiscard]] ImageStatus stdfilt_2d(ArrayRef<f64> img_col_major, usize height, usize width,
                                     ArrayRef<u8> domain_col_major, usize dom_h, usize dom_w,
                                     MutArrayRef<f64> out_col_major, Allocator* alloc = nullptr);
[[nodiscard]] ImageStatus ordfilt2_2d(ArrayRef<f64> img_col_major, usize height, usize width,
                                      i32 nth, ArrayRef<u8> domain_col_major, usize dom_h,
                                      usize dom_w, ArrayRef<f64> additive_offsets, PadMode boundary,
                                      f64 pad_value, MutArrayRef<f64> out_col_major,
                                      Allocator* alloc = nullptr);
[[nodiscard]] ImageStatus entropyfilt_u8(ArrayRef<u8> img_col_major, usize height, usize width,
                                         i32 nbins, ArrayRef<u8> domain_col_major, usize dom_h,
                                         usize dom_w, MutArrayRef<f64> out_col_major,
                                         Allocator* alloc = nullptr);
[[nodiscard]] ImageStatus rangefilt_2d(ArrayRef<f64> img_col_major, usize height, usize width,
                                       ArrayRef<u8> domain_col_major, usize dom_h, usize dom_w,
                                       MutArrayRef<f64> out_col_major, Allocator* alloc = nullptr);

// Adaptive 2D Wiener filter (`noise_in` NaN => estimate as mean local variance).
[[nodiscard]] ImageStatus wiener2_2d(ArrayRef<f64> unit_col_major, usize height, usize width,
                                     usize nh, usize nw, f64 noise_in,
                                     MutArrayRef<f64> out_col_major, f64& noise_out,
                                     Allocator* alloc = nullptr);

// 3D separable box, separable Gaussian (replicate border), and 3D median (symmetric border).
[[nodiscard]] ImageStatus imboxfilt3_f64(ArrayRef<f64> vol_col_major, usize height, usize width,
                                         usize pages, i32 fh, i32 fw, i32 fp,
                                         MutArrayRef<f64> out_col_major,
                                         Allocator* alloc = nullptr);
[[nodiscard]] ImageStatus imgaussfilt3_f64(ArrayRef<f64> vol_col_major, usize height, usize width,
                                           usize pages, f64 sig_h, f64 sig_w, f64 sig_p,
                                           MutArrayRef<f64> out_col_major,
                                           Allocator* alloc = nullptr);
[[nodiscard]] ImageStatus medfilt3_f64(ArrayRef<f64> vol_col_major, usize height, usize width,
                                       usize pages, i32 m, i32 n, i32 p,
                                       MutArrayRef<f64> out_col_major, Allocator* alloc = nullptr);

// 2D convolution Toeplitz matrix (`((m+h_rows-1)*(n+h_cols-1)) x (m*n)`).
[[nodiscard]] ImageStatus convmtx2_f64(ArrayRef<f64> h_col_major, usize h_rows, usize h_cols,
                                       usize m, usize n, MutArrayRef<f64> out_col_major) noexcept;

// 2D frequency response (`out_h` is `m x n` col-major, `out_f1` length `m`, `out_f2` length `n`).
[[nodiscard]] ImageStatus freqz2_2d(ArrayRef<f64> h_col_major, usize h_rows, usize h_cols, usize m,
                                    usize n, MutArrayRef<c64> out_h_col_major,
                                    MutArrayRef<f64> out_f1, MutArrayRef<f64> out_f2,
                                    Allocator* alloc = nullptr);

// Perona-Malik 2D anisotropic diffusion.
[[nodiscard]] ImageStatus imdiffusefilt_2d(ArrayRef<f64> img_col_major, usize height, usize width,
                                           ArrayRef<f64> grad_thresh_per_iter, bool maximal_conn,
                                           bool exponential_cond, MutArrayRef<f64> out_col_major,
                                           Allocator* alloc = nullptr);

// Single-filter 2D Gabor magnitude and phase (FFT domain).
[[nodiscard]] ImageStatus imgaborfilt_2d(ArrayRef<f64> img_col_major, usize height, usize width,
                                         f64 wavelength, f64 orientation_deg, f64 sfb, f64 aspect,
                                         MutArrayRef<f64> out_mag_col_major,
                                         MutArrayRef<f64> out_phase_col_major,
                                         Allocator* alloc = nullptr);

// Immerkaer (1996) noise standard deviation estimate & Non-Local Means 2D denoising.
[[nodiscard]] f64 immerkaer_dos_2d(ArrayRef<f64> img_col_major, usize height, usize width) noexcept;
[[nodiscard]] ImageStatus imnlmfilt_2d(ArrayRef<f64> img_col_major, usize height, usize width,
                                       f64 dos, i32 swsize, i32 cwsize,
                                       MutArrayRef<f64> out_col_major, f64& est_dos_out,
                                       Allocator* alloc = nullptr);

// ============================================================================
// Image Transforms, Test Patterns, Correlation & Deconvolution
// ============================================================================

enum class PhantomModel : u8 {
    kModifiedSheppLogan = 0,
    kSheppLogan,
};

// Orthonormal N x N Type-II DCT basis matrix (`dctmtx`).
[[nodiscard]] ImageStatus dct_matrix(usize n, MutArrayRef<f64> out_col_major) noexcept;

// Separable 2D Type-II DCT and Type-III IDCT (`height x width` col-major).
[[nodiscard]] ImageStatus dct_2d(ArrayRef<f64> in_col_major, usize height, usize width,
                                 MutArrayRef<f64> out_col_major, Allocator* alloc = nullptr);
[[nodiscard]] ImageStatus idct_2d(ArrayRef<f64> in_col_major, usize height, usize width,
                                  MutArrayRef<f64> out_col_major, Allocator* alloc = nullptr);

// 2D and 3D integral images (zero-bordered `(h+1) x (w+1) [x (d+1)]` col-major).
[[nodiscard]] ImageStatus integral_image_2d(ArrayRef<f64> in_col_major, usize height, usize width,
                                            MutArrayRef<f64> out_sat_col_major) noexcept;
[[nodiscard]] ImageStatus integral_image_3d(ArrayRef<f64> in_col_major, usize height, usize width,
                                            usize depth,
                                            MutArrayRef<f64> out_sat_col_major) noexcept;

// Standard 10 x 6 Shepp-Logan ellipse table and `n x n` phantom rasterizer.
[[nodiscard]] ImageStatus phantom_ellipses(PhantomModel model,
                                           MutArrayRef<f64> out_10x6_col_major) noexcept;
[[nodiscard]] ImageStatus phantom_from_ellipses(ArrayRef<f64> ellipses_col_major,
                                                usize num_ellipses, usize n,
                                                MutArrayRef<f64> out_col_major,
                                                Allocator* alloc = nullptr);

// Checkerboard test image (`(2*tiles_y*side) x (2*tiles_x*side)` col-major).
[[nodiscard]] ImageStatus checkerboard(usize side, usize tiles_y, usize tiles_x,
                                       MutArrayRef<f64> out_col_major,
                                       Allocator* alloc = nullptr);

// Point-spread function <-> optical transfer function conversions.
[[nodiscard]] ImageStatus psf_to_otf(ArrayRef<f64> psf_col_major, usize in_h, usize in_w,
                                     usize out_h, usize out_w, MutArrayRef<c64> out_otf_col_major,
                                     Allocator* alloc = nullptr);
[[nodiscard]] ImageStatus otf_to_psf(ArrayRef<c64> otf_col_major, usize in_h, usize in_w,
                                     usize out_h, usize out_w, MutArrayRef<c64> out_psf_col_major,
                                     Allocator* alloc = nullptr);

// Normalized 2D cross-correlation (`(ih+th-1) x (iw+tw-1)` col-major).
[[nodiscard]] ImageStatus normxcorr2(ArrayRef<f64> templ_col_major, usize th, usize tw,
                                     ArrayRef<f64> img_col_major, usize ih, usize iw,
                                     MutArrayRef<f64> out_col_major, Allocator* alloc = nullptr);

// 2D Wiener deconvolution (`ncorr`/`icorr` size 1 for scalar spectrum or `height*width` ACF).
[[nodiscard]] ImageStatus deconvwnr_2d(ArrayRef<f64> img_col_major, usize height, usize width,
                                       ArrayRef<f64> psf_col_major, usize psf_h, usize psf_w,
                                       ArrayRef<f64> ncorr_col_major, ArrayRef<f64> icorr_col_major,
                                       MutArrayRef<f64> out_col_major, Allocator* alloc = nullptr);

// 2D edge taper for FFT-based deconvolution.
[[nodiscard]] ImageStatus edgetaper_2d(ArrayRef<f64> img_col_major, usize height, usize width,
                                       ArrayRef<f64> psf_col_major, usize psf_h, usize psf_w,
                                       MutArrayRef<f64> out_col_major, Allocator* alloc = nullptr);

// 2D Tikhonov-regularized deconvolution (`regop_h == 0 && regop_w == 0` uses default 3x3 Laplacian).
[[nodiscard]] ImageStatus deconvreg_2d(ArrayRef<f64> img_col_major, usize height, usize width,
                                       ArrayRef<f64> psf_col_major, usize psf_h, usize psf_w,
                                       f64 noise_power, f64 lambda_lo, f64 lambda_hi,
                                       ArrayRef<f64> regop_col_major, usize regop_h, usize regop_w,
                                       MutArrayRef<f64> out_col_major, f64& lambda_out,
                                       Allocator* alloc = nullptr);

// ============================================================================
// Connected Components, Distance Transform & Region Descriptors
// ============================================================================

inline constexpr u32 kRegionPropBasic = 0u;
inline constexpr u32 kRegionPropEllipse = 1u << 0;
inline constexpr u32 kRegionPropPerimeter = 1u << 1;

struct RegionProps2D {
    i64 area = 0;
    f64 centroid_x = 0.0; // 0-based column mean
    f64 centroid_y = 0.0; // 0-based row mean
    i32 min_col = 0;
    i32 min_row = 0;
    i32 max_col = 0;
    i32 max_row = 0;
    f64 major_axis_length = 0.0;
    f64 minor_axis_length = 0.0;
    f64 eccentricity = 0.0;
    f64 orientation_deg = 0.0;
    f64 equiv_diameter = 0.0;
    f64 extent = 0.0;
    f64 perimeter = 0.0;
};

enum class RegionAttr : u8 {
    kArea = 0,
    kCircularity,
    kConvexArea,
    kEccentricity,
    kEquivDiameter,
    kEulerNumber,
    kExtent,
    kFilledArea,
    kMajorAxisLength,
    kMaxIntensity,
    kMeanIntensity,
    kMinIntensity,
    kMinorAxisLength,
    kOrientation,
    kPerimeter,
    kPerimeterOld,
    kSolidity,
};

// 2D / 3D connected component labeling (column-major scan order, labels 1..num_labels_out).
[[nodiscard]] ImageStatus bwlabel_2d(ArrayRef<u8> bw_col_major, usize height, usize width, i32 conn,
                                     MutArrayRef<i32> labels_col_major, i32& num_labels_out,
                                     Allocator* alloc = nullptr);
[[nodiscard]] ImageStatus bwlabel_3d(ArrayRef<u8> vol_col_major, usize height, usize width,
                                     usize depth, i32 conn, MutArrayRef<i32> labels_col_major,
                                     i32& num_labels_out, Allocator* alloc = nullptr);

// Binary perimeter, area opening, area filter, seed-based component selection, Euler number.
[[nodiscard]] ImageStatus bwperim_2d(ArrayRef<u8> bw_col_major, usize height, usize width, i32 conn,
                                     MutArrayRef<u8> out_col_major) noexcept;
[[nodiscard]] ImageStatus bwareaopen_2d(ArrayRef<u8> bw_col_major, usize height, usize width,
                                        i32 min_area, i32 conn, MutArrayRef<u8> out_col_major,
                                        Allocator* alloc = nullptr);
[[nodiscard]] ImageStatus bwareafilt_2d(ArrayRef<u8> bw_col_major, usize height, usize width,
                                        f64 lo, f64 hi, usize n_keep, bool keep_largest, i32 conn,
                                        MutArrayRef<u8> out_col_major, Allocator* alloc = nullptr);
[[nodiscard]] ImageStatus bwselect_2d(ArrayRef<u8> bw_col_major, usize height, usize width,
                                      ArrayRef<i32> seed_rows, ArrayRef<i32> seed_cols, i32 conn,
                                      MutArrayRef<u8> out_col_major, Allocator* alloc = nullptr);
[[nodiscard]] ImageStatus bweuler_2d(ArrayRef<u8> bw_col_major, usize height, usize width, i32 conn,
                                     f64& euler_out) noexcept;

// Exact O(HW) Euclidean distance transform (Felzenszwalb-Huttenlocher 2004).
[[nodiscard]] ImageStatus bwdist_euclidean_2d(ArrayRef<u8> bw_col_major, usize height, usize width,
                                              MutArrayRef<f64> out_dist_col_major,
                                              Allocator* alloc = nullptr);

// Moore-Neighbor outer boundary tracing for all connected components (CSR offsets).
[[nodiscard]] ImageStatus bwboundaries_2d(ArrayRef<u8> bw_col_major, usize height, usize width,
                                          i32 conn, MutArrayRef<i32> labels_col_major,
                                          i32& num_labels_out, Array<BoundaryPoint>& points_out,
                                          Array<usize>& offsets_out, Allocator* alloc = nullptr);

// Vossepoel-Smeulders boundary perimeter for a single labeled region.
[[nodiscard]] f64 region_perimeter_2d(ArrayRef<i32> labels_col_major, usize height, usize width,
                                      i32 label, i32 start_row, i32 start_col,
                                      Allocator* alloc = nullptr);

// Per-region geometric descriptors from a 1..num_labels label image.
[[nodiscard]] ImageStatus region_props_2d(ArrayRef<i32> labels_col_major, usize height, usize width,
                                          i32 num_labels, u32 prop_flags,
                                          Array<RegionProps2D>& out_props,
                                          Allocator* alloc = nullptr);

// Component attribute evaluation (supports all 17 bwpropfilt attributes on CSR pixel lists).
[[nodiscard]] ImageStatus region_component_attrs(ArrayRef<i32> pixel_rows, ArrayRef<i32> pixel_cols,
                                                 ArrayRef<usize> comp_offsets, usize height,
                                                 usize width, i32 conn, RegionAttr attr,
                                                 ArrayRef<f64> marker_col_major,
                                                 MutArrayRef<f64> out_values,
                                                 Allocator* alloc = nullptr);

// 8-direction Freeman chain code and cyclic mod-8 first difference.
[[nodiscard]] ImageStatus freeman_chain_code(ArrayRef<f64> rows, ArrayRef<f64> cols,
                                             Array<f64>& out_fcc, Array<f64>& out_diff);

// ============================================================================
// Gradients, Edge Detection, Corners & Hough Transforms
// ============================================================================

enum class GradientMethod : u8 {
    kSobel = 0,
    kPrewitt,
    kCentral,
    kIntermediate,
};

enum class EdgeMethod : u8 {
    kSobel = 0,
    kPrewitt,
    kRoberts,
    kLog,
    kZeroCross,
    kCanny,
};

enum class CornerMethod : u8 {
    kHarris = 0,
    kMinimumEigenvalue,
};

struct HoughLine {
    f64 p1x = 0.0; // 0-based column
    f64 p1y = 0.0; // 0-based row
    f64 p2x = 0.0; // 0-based column
    f64 p2y = 0.0; // 0-based row
    f64 theta = 0.0;
    f64 rho = 0.0;
};

enum class CirclePolarity : u8 {
    kBright = 0,
    kDark,
};

enum class CircleMethod : u8 {
    kPhaseCode = 0,
    kTwoStage,
};

struct CircleDetection {
    f64 center_x = 0.0; // 0-based column
    f64 center_y = 0.0; // 0-based row
    f64 radius = 0.0;
    f64 metric = 0.0;
};

// 2D and 3D image directional gradients (`height x width [x depth]` col-major).
[[nodiscard]] ImageStatus imgradientxy_2d(ArrayRef<f64> img_col_major, usize height, usize width,
                                          GradientMethod method, MutArrayRef<f64> out_gx_col_major,
                                          MutArrayRef<f64> out_gy_col_major) noexcept;
[[nodiscard]] ImageStatus imgradientxyz_3d(ArrayRef<f64> vol_col_major, usize height, usize width,
                                           usize depth, GradientMethod method,
                                           MutArrayRef<f64> out_gx, MutArrayRef<f64> out_gy,
                                           MutArrayRef<f64> out_gz) noexcept;

// 2D edge detection (`thresh` NaN => automatic threshold).
[[nodiscard]] ImageStatus edge_2d(ArrayRef<f64> img_col_major, usize height, usize width,
                                  EdgeMethod method, f64 thresh, MutArrayRef<u8> out_col_major,
                                  Allocator* alloc = nullptr);

// Harris / Shi-Tomasi corner response matrix (`filter_coef` empty => default 5-tap Gaussian).
[[nodiscard]] ImageStatus cornermetric_2d(ArrayRef<f64> unit_col_major, usize height, usize width,
                                          CornerMethod method, f64 sensitivity_factor,
                                          ArrayRef<f64> filter_coef, MutArrayRef<f64> out_col_major,
                                          Allocator* alloc = nullptr);

// Extract up to `max_n` (`< 0` for all) 0-based corner peaks from a cornermetric map.
[[nodiscard]] ImageStatus corner_peaks_2d(ArrayRef<f64> metric_col_major, usize height, usize width,
                                          i32 max_n, f64 quality_level,
                                          Array<BoundaryPoint>& out_peaks,
                                          Allocator* alloc = nullptr);

// Standard 2D Hough transform, peak extraction, and line segment extraction.
[[nodiscard]] ImageStatus hough_2d(ArrayRef<u8> bw_col_major, usize height, usize width,
                                   f64 rho_res, ArrayRef<f64> theta_deg,
                                   Array<f64>& out_h_col_major, Array<f64>& out_rho,
                                   Allocator* alloc = nullptr);
[[nodiscard]] ImageStatus houghpeaks_2d(ArrayRef<f64> h_col_major, usize nrho, usize ntheta,
                                        usize num_peaks, f64 threshold, usize nhood_rho,
                                        usize nhood_theta, ArrayRef<f64> theta_deg,
                                        Array<BoundaryPoint>& out_peaks,
                                        Allocator* alloc = nullptr);
[[nodiscard]] ImageStatus houghlines_2d(ArrayRef<u8> bw_col_major, usize height, usize width,
                                        ArrayRef<f64> theta_deg, ArrayRef<f64> rho,
                                        ArrayRef<BoundaryPoint> peaks_0based, f64 fill_gap,
                                        f64 min_length, Array<HoughLine>& out_lines,
                                        Allocator* alloc = nullptr);

// Circular Hough Transform (`edge_threshold < 0` => automatic Otsu threshold on normalized Sobel).
[[nodiscard]] ImageStatus imfindcircles_2d(ArrayRef<f64> gray_col_major, usize height, usize width,
                                           f64 rmin, f64 rmax, CirclePolarity polarity,
                                           CircleMethod method, f64 sensitivity, f64 edge_threshold,
                                           bool use_single_grad,
                                           Array<CircleDetection>& out_circles,
                                           Allocator* alloc = nullptr);

// ============================================================================
// Segmentation, Geodesic Distances, Inpainting & Polyline Simplification
// ============================================================================

enum class GeodesicMetric : u8 {
    kCityblock = 0,
    kChessboard,
    kQuasiEuclidean,
};

// Binary mask similarity metrics.
[[nodiscard]] f64 dice_coeff(ArrayRef<u8> a, ArrayRef<u8> b) noexcept;
[[nodiscard]] f64 jaccard_index(ArrayRef<u8> a, ArrayRef<u8> b) noexcept;

// Region boundary mask from a 2D label matrix.
[[nodiscard]] ImageStatus boundarymask_2d(ArrayRef<i64> labels_col_major, usize height, usize width,
                                          i32 conn, MutArrayRef<u8> out_col_major) noexcept;

// 8-connected intensity-tolerance region growing from 0-based `(seed_row, seed_col)`.
[[nodiscard]] ImageStatus grayconnected_2d(ArrayRef<f64> img_col_major, usize height, usize width,
                                           i32 seed_row, i32 seed_col, f64 tol,
                                           MutArrayRef<u8> out_col_major,
                                           Allocator* alloc = nullptr);

// Fast-marching pixel weights from intensity difference or DoG gradient magnitude.
[[nodiscard]] ImageStatus graydiffweight_f64(ArrayRef<f64> img, f64 ref_gray_val,
                                             f64 rolloff_factor, f64 cutoff,
                                             MutArrayRef<f64> out_weights) noexcept;
[[nodiscard]] ImageStatus gradientweight_2d(ArrayRef<f64> img_col_major, usize height, usize width,
                                            f64 sigma_x, f64 sigma_y, f64 rolloff_factor,
                                            f64 weight_cutoff,
                                            MutArrayRef<f64> out_weights_col_major,
                                            Allocator* alloc = nullptr);

// Harmonic (Laplacian) region inpainting solved by conjugate gradient.
[[nodiscard]] ImageStatus regionfill_2d(ArrayRef<f64> img_col_major, ArrayRef<u8> mask_col_major,
                                        usize height, usize width, MutArrayRef<f64> out_col_major,
                                        Allocator* alloc = nullptr);

// Scanline even-odd polygon-to-mask rasterizer (1-based vertex coordinates).
[[nodiscard]] ImageStatus poly2mask_2d(ArrayRef<f64> vx_1based, ArrayRef<f64> vy_1based,
                                       usize height, usize width, MutArrayRef<u8> out_bw_col_major,
                                       Allocator* alloc = nullptr);

// Gray-weighted and binary geodesic distance transforms (Dijkstra).
[[nodiscard]] ImageStatus graydist_2d(ArrayRef<f64> img_col_major, usize height, usize width,
                                      ArrayRef<usize> seed_indices_0based, GeodesicMetric metric,
                                      MutArrayRef<f64> out_dist_col_major,
                                      Allocator* alloc = nullptr);
[[nodiscard]] ImageStatus bwdistgeodesic_2d(ArrayRef<u8> bw_col_major, usize height, usize width,
                                            ArrayRef<usize> seed_indices_0based,
                                            GeodesicMetric metric,
                                            MutArrayRef<f64> out_dist_col_major,
                                            Allocator* alloc = nullptr);

// Ramer-Douglas-Peucker polyline simplification (returns kept 0-based vertex indices).
[[nodiscard]] ImageStatus reducepoly_f64(ArrayRef<f64> xs, ArrayRef<f64> ys, f64 tolerance,
                                         Array<i32>& out_kept_indices);
[[nodiscard]] ImageStatus reducepoly_f32(ArrayRef<f32> xs, ArrayRef<f32> ys, f64 tolerance,
                                         Array<i32>& out_kept_indices);

// 2D Meyer watershed transform (0 = ridge line, 1..num_basins_out = catchment basins).
[[nodiscard]] ImageStatus watershed_2d(ArrayRef<f64> img_col_major, usize height, usize width,
                                       i32 conn, MutArrayRef<u32> out_labels_col_major,
                                       u32& num_basins_out, Allocator* alloc = nullptr);

} // namespace keel::image
