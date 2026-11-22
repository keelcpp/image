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

void rotate_2d_dims(usize in_h, usize in_w, f64 angle_deg, bool crop, usize& out_h,
                    usize& out_w) noexcept;

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

} // namespace keel::image
