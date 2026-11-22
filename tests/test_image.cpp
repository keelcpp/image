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

void test_illum_geom_fir2d_dehaze() {
    // 1. illumwhite / illumgray / illumpca / delta_e_94 / delta_e_2000
    const f64 r[4] = {0.2, 0.4, 0.6, 0.8};
    const f64 g[4] = {0.1, 0.2, 0.3, 0.4};
    const f64 b[4] = {0.05, 0.1, 0.15, 0.2};
    im::Color3 iw{}, ig{}, ip{};
    CHECK(im::illumwhite_planar(r, g, b, {}, 0.0, iw) == im::ImageStatus::kOk);
    CHECK_NEAR(iw.c0, 0.8, 1e-12);
    CHECK_NEAR(iw.c1, 0.4, 1e-12);
    CHECK_NEAR(iw.c2, 0.2, 1e-12);

    CHECK(im::illumgray_planar(r, g, b, {}, 0.0, 0.0, 1.0, ig) == im::ImageStatus::kOk);
    CHECK_NEAR(ig.c0, 0.5, 1e-12);
    CHECK_NEAR(ig.c1, 0.25, 1e-12);
    CHECK_NEAR(ig.c2, 0.125, 1e-12);

    CHECK(im::illumpca_planar(r, g, b, {}, 50.0, ip) == im::ImageStatus::kOk);
    // Direction should be proportional to (4, 2, 1) / sqrt(21).
    CHECK_NEAR(ip.c0 / ip.c2, 4.0, 1e-6);
    CHECK_NEAR(ip.c1 / ip.c2, 2.0, 1e-6);

    const im::Color3 lab1{50.0, 2.6772, -79.7751};
    const im::Color3 lab2{50.0, 0.0, -82.7485};
    CHECK_NEAR(im::delta_e_94(lab1, lab1), 0.0, 1e-12);
    CHECK_NEAR(im::delta_e_2000(lab1, lab1), 0.0, 1e-12);
    // Sharma-Wu-Dalal (2005) pair 1 reference value ~2.0425.
    CHECK_NEAR(im::delta_e_2000(lab1, lab2), 2.0425, 1e-3);

    // 2. resize_2d / resize_3d / rotate_2d / translate_2d / pyramid_reduce/expand
    f64 img4[4 * 4]{};
    for (usize i = 0; i < 16; ++i) img4[i] = 3.5;
    f64 res8[8 * 8]{};
    CHECK(im::resize_2d(img4, 4, 4, 1, 8, 8, 2.0, 2.0, im::ResizeKernel::kCubic, true, res8) ==
          im::ImageStatus::kOk);
    CHECK_NEAR(res8[0], 3.5, 1e-12);
    CHECK_NEAR(res8[63], 3.5, 1e-12);

    f64 vol2[2 * 2 * 2]{1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0};
    f64 vol4[4 * 4 * 4]{};
    CHECK(im::resize_3d(vol2, 2, 2, 2, 4, 4, 4, 2.0, 2.0, 2.0, im::ResizeKernel::kLanczos2, true,
                        vol4) == im::ImageStatus::kOk);
    CHECK_NEAR(vol4[0], 1.0, 1e-12);

    usize rh = 0, rw = 0;
    im::rotate_2d_dims(3, 5, 90.0, false, rh, rw);
    CHECK(rh == 5 && rw == 3);
    f64 rect[3 * 5]{};
    for (usize i = 0; i < 15; ++i) rect[i] = keel::as<f64>(i + 1);
    f64 rot90[5 * 3]{};
    CHECK(im::rotate_2d(rect, 3, 5, 1, 90.0, true, false, rot90, rh, rw) == im::ImageStatus::kOk);
    CHECK_NEAR(rot90[0], rect[4 * 3 + 0], 1e-12);

    f64 tr_out[4 * 4]{};
    CHECK(im::translate_2d(img4, 4, 4, 1, 0.0, 0.0, tr_out) == im::ImageStatus::kOk);
    CHECK_NEAR(tr_out[5], 3.5, 1e-12);

    f64 pyr_r[2 * 2]{}, pyr_e[7 * 7]{};
    CHECK(im::pyramid_reduce_2d(img4, 4, 4, 1, pyr_r) == im::ImageStatus::kOk);
    CHECK_NEAR(pyr_r[0], 3.5, 1e-12);
    CHECK(im::pyramid_expand_2d(img4, 4, 4, 1, pyr_e) == im::ImageStatus::kOk);
    CHECK(pyr_e[3 * 7 + 3] > 3.0);

    // 3. fsamp2 / ftrans2 / fwind1 / fwind2_real
    f64 hd5[25]{}, h5[25]{};
    for (usize i = 0; i < 25; ++i) hd5[i] = 1.0;
    CHECK(im::fsamp2_real(hd5, 5, 5, h5) == im::ImageStatus::kOk);
    CHECK_NEAR(h5[2 * 5 + 2], 1.0, 1e-12);
    CHECK_NEAR(h5[0], 0.0, 1e-12);

    const f64 b1d[3] = {0.25, 0.5, 0.25};
    keel::Array<f64> ft;
    usize fth = 0, ftw = 0;
    CHECK(im::ftrans2(b1d, {}, 0, 0, ft, fth, ftw) == im::ImageStatus::kOk);
    CHECK(fth == 3 && ftw == 3);
    f64 ft_sum = 0.0;
    for (usize i = 0; i < 9; ++i) ft_sum += ft[i];
    CHECK_NEAR(ft_sum, 1.0, 1e-12);

    const f64 win5[5] = {0.5, 0.8, 1.0, 0.8, 0.5};
    f64 fw1[25]{};
    CHECK(im::fwind1_real(hd5, 5, 5, win5, {}, fw1) == im::ImageStatus::kOk);
    CHECK_NEAR(fw1[2 * 5 + 2], 1.0, 1e-12);

    f64 fw2[25]{};
    CHECK(im::fwind2_real(hd5, 5, 5, win5, 1, 5, fw2) == im::ImageStatus::kOk);
    CHECK_NEAR(fw2[2 * 5 + 2], 1.0, 1e-12);

    // 4. box_filter_2d_replicate / guided_filter_gray / dehaze
    f64 box_in[16]{}, box_out[16]{}, gf_out[16]{};
    for (usize i = 0; i < 16; ++i) box_in[i] = 0.6;
    CHECK(im::box_filter_2d_replicate(box_in, 4, 4, 3, box_out) == im::ImageStatus::kOk);
    CHECK_NEAR(box_out[5], 0.6, 1e-12);
    CHECK(im::guided_filter_gray(box_in, {}, 4, 4, 3, 0.01, gf_out) == im::ImageStatus::kOk);
    CHECK_NEAR(gf_out[5], 0.6, 1e-12);

    keel::Array<f32> hazy(16 * 16 * 3, 0.0f);
    for (usize c = 0; c < 3; ++c) {
        for (usize i = 0; i < 256; ++i) {
            // Simulated haze: scene in [0.1, 0.6] blended with atmospheric light 0.9.
            const f32 scene = 0.1f + 0.5f * keel::as<f32>(i % 16) / 15.0f;
            hazy[c * 256 + i] = 0.5f * scene + 0.5f * 0.9f;
        }
    }
    keel::Array<f32> dehazed(16 * 16 * 3, 0.0f);
    keel::Array<f64> thick(256, 0.0);
    f64 atm_est[3]{};
    CHECK(im::dehaze(hazy.as_ref(), 16, 16, 3, 0.9, im::DehazeMethod::kSimpleDcp, {},
                     im::DehazeContrast::kNone, 0.0, dehazed.as_mut_ref(), thick.as_mut_ref(),
                     atm_est) == im::ImageStatus::kOk);
    CHECK(atm_est[0] > 0.7 && atm_est[0] <= 1.0);
    CHECK(dehazed[0] < hazy[0]); // haze reduced at darkest pixel

    CHECK(im::dehaze(hazy.as_ref(), 16, 16, 3, 0.9, im::DehazeMethod::kApproxDcp, {},
                     im::DehazeContrast::kGlobal, 0.0, dehazed.as_mut_ref(), thick.as_mut_ref(),
                     atm_est) == im::ImageStatus::kOk);
    CHECK(atm_est[0] > 0.7 && atm_est[0] <= 1.0);
}

void test_contrast() {
    // 1. stretchlim_unit / imadjust_unit / histeq_unit / shannon_entropy
    const f64 ramp[10] = {0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0};
    f64 lim[2]{};
    CHECK(im::stretchlim_unit(ramp, 10, 1, 256, 0.01, 0.99, lim) == im::ImageStatus::kOk);
    CHECK(lim[0] < lim[1]);

    f64 adj[10]{};
    CHECK(im::imadjust_unit(ramp, 0.1, 1.0, 0.0, 1.0, 1.0, adj) == im::ImageStatus::kOk);
    CHECK_NEAR(adj[0], 0.0, 1e-12);
    CHECK_NEAR(adj[9], 1.0, 1e-12);

    f64 eq[10]{};
    CHECK(im::histeq_unit(ramp, 64, eq) == im::ImageStatus::kOk);
    CHECK(eq[0] <= eq[9] && eq[9] <= 1.0);

    const f64 h_unif[4] = {10.0, 10.0, 10.0, 10.0};
    CHECK_NEAR(im::shannon_entropy(h_unif), 2.0, 1e-12);

    // 2. clahe_2d (uniform, rayleigh, exponential)
    keel::Array<f64> img(16 * 16, 0.0);
    for (usize c = 0; c < 16; ++c)
        for (usize r = 0; r < 16; ++r)
            img[c * 16 + r] = keel::as<f64>(r + c) / 30.0;
    keel::Array<f64> cl_out(16 * 16, 0.0);
    CHECK(im::clahe_2d(img.as_ref(), 16, 16, 2, 2, 0.01, 64, im::ClaheDistribution::kUniform, 0.4,
                       0.0, 1.0, false, cl_out.as_mut_ref()) == im::ImageStatus::kOk);
    CHECK(cl_out[0] < cl_out[255]);

    // 3. multithresh_otsu (1, 2, 3 levels)
    keel::Array<f64> bimod(40, 0.0);
    for (usize i = 0; i < 10; ++i) bimod[i] = 10.0;
    for (usize i = 10; i < 20; ++i) bimod[i] = 80.0;
    for (usize i = 20; i < 30; ++i) bimod[i] = 160.0;
    for (usize i = 30; i < 40; ++i) bimod[i] = 240.0;
    f64 t1[1]{}, t2[2]{}, t3[3]{}, em = 0.0;
    CHECK(im::multithresh_otsu(bimod.as_ref(), 1, true, t1, em) == im::ImageStatus::kOk);
    CHECK(t1[0] >= 80.0 && t1[0] <= 160.0 && em > 0.7);
    CHECK(im::multithresh_otsu(bimod.as_ref(), 2, true, t2, em) == im::ImageStatus::kOk);
    CHECK(t2[0] < t2[1]);
    CHECK(im::multithresh_otsu(bimod.as_ref(), 3, true, t3, em) == im::ImageStatus::kOk);
    CHECK(t3[0] < t3[1] && t3[1] < t3[2] && em > 0.95);

    // 4. imhistmatch_unit / adaptthresh_2d / flatfield_2d / wcodemat
    f64 hm_out[10]{}, hg[16]{};
    CHECK(im::imhistmatch_unit(ramp, ramp, 256, 16, hm_out, hg) == im::ImageStatus::kOk);
    CHECK(hm_out[0] <= hm_out[9]);

    keel::Array<f64> thr(16 * 16, 0.0);
    CHECK(im::adaptthresh_2d(img.as_ref(), 16, 16, 0.5, 5, false, thr.as_mut_ref()) ==
          im::ImageStatus::kOk);
    CHECK(thr[0] >= 0.0 && thr[255] <= 1.0);
    CHECK(im::adaptthresh_2d(img.as_ref(), 16, 16, 0.5, 5, true, thr.as_mut_ref()) ==
          im::ImageStatus::kOk);

    keel::Array<f64> ff_in(16 * 16, 0.5), ff_out(16 * 16, 0.0);
    CHECK(im::flatfield_2d(ff_in.as_ref(), 16, 16, 1, 3.0, {}, ff_out.as_mut_ref()) ==
          im::ImageStatus::kOk);
    CHECK_NEAR(ff_out[100], 0.5, 1e-10);

    const f64 w_in[6] = {1.0, 4.0, -2.0, -5.0, 3.0, 6.0};
    f64 w_out[6]{};
    CHECK(im::wcodemat(w_in, 2, 3, 1, 4, im::WcodematMode::kMat, true, w_out) ==
          im::ImageStatus::kOk);
    CHECK_NEAR(w_out[0], 1.0, 1e-12);
    CHECK_NEAR(w_out[5], 4.0, 1e-12);
}

} // namespace

int main() {
    test_color();
    test_quality();
    test_threshold();
    test_demosaic_and_texture();
    test_chromadapt_and_filters();
    test_illum_geom_fir2d_dehaze();
    test_contrast();

    if (g_failures == 0) {
        std::printf("ALL keel::image TESTS PASSED\n");
        return 0;
    }
    std::fprintf(stderr, "TOTAL FAILURES: %d\n", g_failures);
    return 1;
}
