#include "keel/image.h"

#include "keel/math.h"

#include <cstdio>

namespace {

using namespace keel;
namespace img = keel::image;

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

} // namespace

int main() {
    test_color();
    test_quality();
    test_threshold();

    if (g_failures == 0) {
        std::printf("ALL keel::image TESTS PASSED\n");
        return 0;
    }
    std::fprintf(stderr, "TOTAL FAILURES: %d\n", g_failures);
    return 1;
}
