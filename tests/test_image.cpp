#include "keel/image.h"

#include "keel/math.h"

#include <cstdio>

namespace {

using namespace keel;
namespace img = keel::image;
namespace im = keel::image;

i32 g_failures = 0;

void check_true(bool cond, const char* expr, i32 line) {
    if (!cond) {
        std::fprintf(stderr, "FAIL line %d: %s\n", line, expr);
        ++g_failures;
    }
}

void check_near(f64 actual, f64 expected, f64 tol, const char* expr, i32 line) {
    if (!(keel::abs(actual - expected) <= tol)) {
        std::fprintf(stderr, "FAIL line %d: %s (actual=%.15g, expected=%.15g, tol=%.3g)\n",
                     line, expr, actual, expected, tol);
        ++g_failures;
    }
}

#define CHECK(cond) check_true((cond), #cond, __LINE__)
#define CHECK_NEAR(a, e, t) check_near((a), (e), (t), #a " ~= " #e, __LINE__)

void test_color() {
    // RGB <-> HSV roundtrip
    const f64 r[4] = {1.0, 0.2, 0.0, 0.5};
    const f64 g[4] = {0.0, 0.7, 0.5, 0.5};
    const f64 b[4] = {0.0, 0.4, 1.0, 0.5};
    f64 c0[4]{}, c1[4]{}, c2[4]{};
    f64 r2[4]{}, g2[4]{}, b2[4]{};

    CHECK(img::rgb_to_hsv_planar(r, g, b, c0, c1, c2) == img::ImageStatus::kOk);
    CHECK(img::hsv_to_rgb_planar(c0, c1, c2, r2, g2, b2) == img::ImageStatus::kOk);
    for (usize i = 0; i < 4; ++i) {
        CHECK_NEAR(r2[i], r[i], 1e-13);
        CHECK_NEAR(g2[i], g[i], 1e-13);
        CHECK_NEAR(b2[i], b[i], 1e-13);
    }

    // RGB <-> YCbCr roundtrip
    CHECK(img::rgb_to_ycbcr_planar(r, g, b, c0, c1, c2) == img::ImageStatus::kOk);
    CHECK(img::ycbcr_to_rgb_planar(c0, c1, c2, r2, g2, b2) == img::ImageStatus::kOk);
    for (usize i = 0; i < 4; ++i) {
        CHECK_NEAR(r2[i], r[i], 1e-12);
        CHECK_NEAR(g2[i], g[i], 1e-12);
        CHECK_NEAR(b2[i], b[i], 1e-12);
    }

    // RGB <-> Lab roundtrip
    CHECK(img::rgb_to_lab_planar(r, g, b, c0, c1, c2) == img::ImageStatus::kOk);
    CHECK(img::lab_to_rgb_planar(c0, c1, c2, r2, g2, b2) == img::ImageStatus::kOk);
    for (usize i = 0; i < 4; ++i) {
        CHECK_NEAR(r2[i], r[i], 1e-5);
        CHECK_NEAR(g2[i], g[i], 1e-5);
        CHECK_NEAR(b2[i], b[i], 1e-5);
    }

    // White D65 -> Lab (100, 0, 0)
    const img::Color3 white_lab = img::rgb_to_lab(1.0, 1.0, 1.0);
    CHECK_NEAR(white_lab.c0, 100.0, 1e-3);
    CHECK_NEAR(white_lab.c1, 0.0, 1e-2);
    CHECK_NEAR(white_lab.c2, 0.0, 1e-2);

    // Grayscale u8 & f64
    f64 gray[4]{};
    CHECK(img::rgb_to_gray_planar(r, g, b, gray) == img::ImageStatus::kOk);
    CHECK_NEAR(gray[3], 0.5, 1e-14);

    const u8 ru[2] = {255, 100};
    const u8 gu[2] = {255, 100};
    const u8 bu[2] = {255, 100};
    u8 gray_u[2]{};
    CHECK(img::rgb_to_gray_u8_planar(ru, gu, bu, gray_u) == img::ImageStatus::kOk);
    CHECK(gray_u[0] == 255);
    CHECK(gray_u[1] == 100);
}

void test_quality() {
    const f64 a[4] = {0.2, 0.4, 0.6, 0.8};
    const f64 b[4] = {0.2, 0.5, 0.6, 0.7};
    CHECK_NEAR(img::mse(a, a), 0.0, 1e-15);
    CHECK_NEAR(img::mse(a, b), 0.005, 1e-15);
    CHECK(keel::is_inf(img::psnr(a, a, 1.0)));
    CHECK_NEAR(img::psnr(a, b, 1.0), 10.0 * keel::log10(1.0 / 0.005), 1e-12);

    CHECK_NEAR(img::mean2(a), 0.5, 1e-15);
    CHECK_NEAR(img::std2(a), keel::sqrt(0.2 / 3.0), 1e-14);

    // Gaussian kernel normalization & symmetry
    f64 k[25]{};
    CHECK(img::gaussian_kernel_2d(5, 5, 1.0, k) == img::ImageStatus::kOk);
    f64 ksum = 0.0;
    for (usize i = 0; i < 25; ++i) ksum += k[i];
    CHECK_NEAR(ksum, 1.0, 1e-14);
    CHECK_NEAR(k[0], k[24], 1e-15);

    // SSIM identical == 1.0, different < 1.0
    constexpr usize H = 16;
    constexpr usize W = 16;
    f64 im1[H * W]{};
    f64 im2[H * W]{};
    for (usize c = 0; c < W; ++c) {
        for (usize r = 0; r < H; ++r) {
            const f64 v = keel::as<f64>(r + c) / keel::as<f64>(H + W - 2);
            im1[c * H + r] = v;
            im2[c * H + r] = keel::clamp(v + 0.1 * ((r & 1u) ? 1.0 : -1.0), 0.0, 1.0);
        }
    }
    CHECK_NEAR(img::ssim(im1, im1, H, W, 1.0), 1.0, 1e-12);
    const f64 s = img::ssim(im1, im2, H, W, 1.0);
    CHECK(s > 0.0 && s < 1.0);
}

void test_threshold() {
    // Bimodal distribution: peaks at bin 20 and bin 200 out of 256
    f64 counts[256]{};
    counts[20] = 500.0;
    counts[200] = 500.0;
    const img::OtsuResult o = img::otsu_threshold(counts);
    // All thresholds t in [20, 199] tie for max sigma_b^2 -> mean is (20 + 199)/2 = 109.5
    CHECK_NEAR(o.bin_level, 109.5, 1e-12);
    CHECK_NEAR(o.threshold, 109.5 / 255.0, 1e-12);
    CHECK_NEAR(o.effectiveness, 1.0, 1e-12);

    // Histogram test
    const f64 vals[4] = {0.0, 0.5, 1.0, 1.0};
    f64 h[3]{};
    CHECK(img::histogram(vals, 0.0, 1.0, h) == img::ImageStatus::kOk);
    CHECK_NEAR(h[0], 1.0, 1e-15);
    CHECK_NEAR(h[1], 1.0, 1e-15);
    CHECK_NEAR(h[2], 2.0, 1e-15);
}

void test_demosaic_and_texture() {
    // Constant Bayer mosaic of value 100 -> constant RGB (100, 100, 100)
    f64 mosaic[6 * 6]{};
    for (usize i = 0; i < 36; ++i) mosaic[i] = 100.0;
    f64 r[36]{}, g[36]{}, b[36]{};
    CHECK(im::demosaic_mhc(mosaic, 6, 6, im::BayerPattern::kRggb, 255.0, r, g, b) ==
          im::ImageStatus::kOk);
    for (usize i = 0; i < 36; ++i) {
        CHECK_NEAR(r[i], 100.0, 1e-12);
        CHECK_NEAR(g[i], 100.0, 1e-12);
        CHECK_NEAR(b[i], 100.0, 1e-12);
    }

    // GLCM & Haralick properties on a 4x4 checkerboard
    f64 chk[16]{};
    for (usize c = 0; c < 4; ++c)
        for (usize r_i = 0; r_i < 4; ++r_i)
            chk[r_i + c * 4] = ((r_i + c) % 2 == 0) ? 0.0 : 1.0;
    f64 glcm[4]{};
    CHECK(im::graycomatrix(chk, 4, 4, 2, 0, 1, 0.0, 1.0, true, glcm) == im::ImageStatus::kOk);
    CHECK_NEAR(glcm[0], 0.0, 1e-12);
    CHECK_NEAR(glcm[1], 12.0, 1e-12);
    CHECK_NEAR(glcm[2], 12.0, 1e-12);
    CHECK_NEAR(glcm[3], 0.0, 1e-12);

    im::GlcmProps props;
    CHECK(im::graycoprops(glcm, 2, props) == im::ImageStatus::kOk);
    CHECK_NEAR(props.contrast, 1.0, 1e-12);
    CHECK_NEAR(props.correlation, -1.0, 1e-12);
    CHECK_NEAR(props.energy, 0.5, 1e-12);
    CHECK_NEAR(props.homogeneity, 0.5, 1e-12);
}

void test_chromadapt_and_filters() {
    // Chromatic adaptation with white illuminant [1,1,1] is identity in sRGB
    const f64 in_r[2] = {0.2, 0.8};
    const f64 in_g[2] = {0.4, 0.6};
    const f64 in_b[2] = {0.7, 0.3};
    f64 out_r[2]{}, out_g[2]{}, out_b[2]{};
    CHECK(im::chromadapt_planar(in_r, in_g, in_b, {1.0, 1.0, 1.0}, im::ChromAdaptMethod::kBradford,
                                im::RgbColorSpace::kSrgb, out_r, out_g, out_b) ==
          im::ImageStatus::kOk);
    for (usize i = 0; i < 2; ++i) {
        CHECK_NEAR(out_r[i], in_r[i], 1e-6);
        CHECK_NEAR(out_g[i], in_g[i], 1e-6);
        CHECK_NEAR(out_b[i], in_b[i], 1e-6);
    }

    // 3D filter kernels: Gaussian sums to 1, Laplacian & LoG sum to 0
    f64 g3[27]{}, l3[27]{}, log3[27]{}, s3[27]{};
    CHECK(im::fspecial3_gaussian(3, 3, 3, 1.0, 1.0, 1.0, g3) == im::ImageStatus::kOk);
    CHECK(im::fspecial3_laplacian(0.2, 0.1, l3) == im::ImageStatus::kOk);
    CHECK(im::fspecial3_log(3, 3, 3, 1.0, 1.0, 1.0, log3) == im::ImageStatus::kOk);
    CHECK(im::fspecial3_gradient(true, im::Axis3D::kX, s3) == im::ImageStatus::kOk);
    f64 sg = 0.0, sl = 0.0, slog = 0.0;
    for (usize i = 0; i < 27; ++i) {
        sg += g3[i];
        sl += l3[i];
        slog += log3[i];
    }
    CHECK_NEAR(sg, 1.0, 1e-12);
    CHECK_NEAR(sl, 0.0, 1e-12);
    CHECK_NEAR(slog, 0.0, 1e-12);

    keel::Array<f64> ell;
    usize eh = 0, ew = 0, ep = 0;
    CHECK(im::fspecial3_ellipsoid(2.0, 2.0, 2.0, ell, eh, ew, ep) == im::ImageStatus::kOk);
    CHECK(eh == 5 && ew == 5 && ep == 5);

    // fwind2 unit DC gain
    f64 hd[9]{}, win[9]{}, h2[9]{};
    for (usize i = 0; i < 9; ++i) {
        hd[i] = 1.0;
        win[i] = 1.0;
    }
    CHECK(im::fwind2(hd, win, 3, 3, h2) == im::ImageStatus::kOk);
    f64 sh2 = 0.0;
    for (usize i = 0; i < 9; ++i) sh2 += h2[i];
    CHECK_NEAR(sh2, 1.0, 1e-12);

    // Fibermetric 2D detects a bright vertical line in the middle
    f64 line_img[16 * 16]{};
    for (usize r = 0; r < 16; ++r) line_img[r + 8 * 16] = 1.0;
    const f64 thick[1] = {4.0};
    f64 vessel[16 * 16]{};
    CHECK(im::fibermetric_2d(line_img, 16, 16, thick, 0.01, true, vessel) == im::ImageStatus::kOk);
    CHECK(vessel[8 + 8 * 16] > 0.5);
    CHECK(vessel[8 + 0 * 16] < 0.05);

    // Local Laplacian filter: identity (alpha=1, beta=1) and detail-preserving on ramp
    f32 ramp[16 * 16]{}, llf_out[16 * 16]{};
    for (usize c = 0; c < 16; ++c)
        for (usize r = 0; r < 16; ++r)
            ramp[r + c * 16] = keel::as<f32>(c) / 15.0f;
    CHECK(im::local_laplacian_gray(ramp, 16, 16, 0.2f, 0.5f, 1.0f, 8, llf_out) ==
          im::ImageStatus::kOk);
    CHECK(llf_out[8 + 15 * 16] > llf_out[8 + 0 * 16]);
}

} // namespace

int main() {
    test_color();
    test_quality();
    test_threshold();
    test_demosaic_and_texture();
    test_chromadapt_and_filters();

    if (g_failures == 0) {
        std::printf("ALL keel::image TESTS PASSED\n");
        return 0;
    }
    std::fprintf(stderr, "TOTAL FAILURES: %d\n", g_failures);
    return 1;
}
