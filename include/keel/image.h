#pragma once

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

} // namespace keel::image
