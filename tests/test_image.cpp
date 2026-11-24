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

    // rotate_2d Keys bicubic (loose and crop)
    f64 blk[10 * 12]{};
    for (usize col = 3; col <= 8; ++col) {
        for (usize row = 2; row <= 6; ++row) {
            blk[col * 10 + row] = 1.0;
        }
    }
    usize bh = 0, bw = 0;
    im::rotate_2d_dims(10, 12, 10.0, false, bh, bw);
    CHECK(bh == 12 && bw == 14);
    f64 rot_bc[12 * 14]{};
    CHECK(im::rotate_2d(blk, 10, 12, 1, 10.0, im::RotateInterp::kBicubic, false, rot_bc, bh, bw) ==
          im::ImageStatus::kOk);
    f64 bc_sum = 0.0;
    for (usize i = 0; i < bh * bw; ++i) bc_sum += rot_bc[i];
    CHECK_NEAR(bc_sum, 30.000512049392, 1e-9);
    CHECK_NEAR(rot_bc[4 * bh + 3], 0.670062299259, 1e-9);
    CHECK_NEAR(rot_bc[5 * bh + 4], 1.021533003648, 1e-9);

    f64 rot_bc_crop[10 * 12]{};
    CHECK(im::rotate_2d(blk, 10, 12, 1, 10.0, im::RotateInterp::kBicubic, true, rot_bc_crop, 10,
                        12) == im::ImageStatus::kOk);
    CHECK_NEAR(rot_bc_crop[4 * 10 + 3], 1.021533003648, 1e-9);
    CHECK_NEAR(rot_bc_crop[5 * 10 + 4], 1.0, 1e-12);

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

void test_morph() {
    // 1. Strel shapes
    im::StrelMask sq, dia, dsk, ln;
    CHECK(im::strel_square(3, sq) == im::ImageStatus::kOk);
    CHECK(sq.rows == 3 && sq.cols == 3 && sq.mask_col_major.size() == 9);
    CHECK(im::strel_diamond(2, dia) == im::ImageStatus::kOk);
    CHECK(dia.rows == 5 && dia.cols == 5 && dia.mask_col_major[0] == 0 &&
          dia.mask_col_major[12] == 1);
    CHECK(im::strel_disk(5.0, 4, dsk) == im::ImageStatus::kOk);
    CHECK(dsk.rows == 9 && dsk.cols == 9 && dsk.mask_col_major[4 * 9 + 4] == 1);
    im::StrelMask dsk_euc;
    CHECK(im::strel_disk(5.0, 0, dsk_euc) == im::ImageStatus::kOk);
    CHECK(dsk_euc.rows == 11 && dsk_euc.cols == 11 && dsk_euc.mask_col_major[5 * 11 + 5] == 1);
    CHECK(im::strel_line(5.0, 0.0, ln) == im::ImageStatus::kOk);
    CHECK(ln.rows == 1 && ln.cols == 5);

    // 2. Erode / dilate (van Herk u8 + f64)
    keel::Array<u8> bw(9 * 9, u8{0});
    bw[4 * 9 + 4] = 1;
    keel::Array<u8> dil(9 * 9, u8{0}), ero(9 * 9, u8{0});
    CHECK(im::morph_dilate_u8(bw.as_ref(), 9, 9, sq.mask_col_major.as_ref(), sq.rows, sq.cols,
                              dil.as_mut_ref()) == im::ImageStatus::kOk);
    usize dil_sum = 0;
    for (usize i = 0; i < 81; ++i) dil_sum += dil[i];
    CHECK(dil_sum == 9);
    CHECK(im::morph_erode_u8(dil.as_ref(), 9, 9, sq.mask_col_major.as_ref(), sq.rows, sq.cols,
                             ero.as_mut_ref()) == im::ImageStatus::kOk);
    CHECK(ero[4 * 9 + 4] == 1);

    keel::Array<f64> gf(9 * 9, 0.0), gd(9 * 9, 0.0), ge(9 * 9, 0.0);
    gf[4 * 9 + 4] = 10.0;
    CHECK(im::morph_dilate_f64(gf.as_ref(), 9, 9, sq.mask_col_major.as_ref(), sq.rows, sq.cols,
                               gd.as_mut_ref()) == im::ImageStatus::kOk);
    CHECK_NEAR(gd[3 * 9 + 3], 10.0, 1e-12);
    CHECK(im::morph_erode_f64(gd.as_ref(), 9, 9, sq.mask_col_major.as_ref(), sq.rows, sq.cols,
                              ge.as_mut_ref()) == im::ImageStatus::kOk);
    CHECK_NEAR(ge[4 * 9 + 4], 10.0, 1e-12);

    // 3. imfill_holes_binary & imclearborder_binary
    keel::Array<u8> ring(7 * 7, u8{0});
    for (usize c = 1; c <= 5; ++c) {
        for (usize r = 1; r <= 5; ++r) {
            if (r == 1 || r == 5 || c == 1 || c == 5) ring[c * 7 + r] = 1;
        }
    }
    keel::Array<u8> filled(7 * 7, u8{0});
    CHECK(im::imfill_holes_binary(ring.as_ref(), 7, 7, 4, filled.as_mut_ref()) ==
          im::ImageStatus::kOk);
    CHECK(filled[3 * 7 + 3] == 1);

    keel::Array<u8> cleared(7 * 7, u8{0});
    ring[0] = 1; // border touch
    CHECK(im::imclearborder_binary(ring.as_ref(), 7, 7, 4, cleared.as_mut_ref()) ==
          im::ImageStatus::kOk);
    CHECK(cleared[0] == 0 && cleared[1 * 7 + 1] == 1);

    // 4. bwpack / bwunpack roundtrip
    keel::Array<u32> packed(1 * 7, u32{0});
    keel::Array<u8> unpacked(7 * 7, u8{0});
    CHECK(im::bwpack_2d(filled.as_ref(), 7, 7, packed.as_mut_ref()) == im::ImageStatus::kOk);
    CHECK(im::bwunpack_2d(packed.as_ref(), 1, 7, 7, unpacked.as_mut_ref()) == im::ImageStatus::kOk);
    for (usize i = 0; i < 49; ++i) CHECK(unpacked[i] == filled[i]);

    // 5. bwmorph & bwmorph3 & bwtraceboundary
    keel::Array<u8> thinned(7 * 7, u8{0});
    CHECK(im::bwmorph_2d(filled.as_ref(), 7, 7, im::BwmorphOp::kThin, -1, thinned.as_mut_ref()) ==
          im::ImageStatus::kOk);
    CHECK(thinned[3 * 7 + 3] == 1);

    keel::Array<u8> vol(3 * 3 * 3, u8{1}), vol_out(3 * 3 * 3, u8{0});
    CHECK(im::bwmorph3_3d(vol.as_ref(), 3, 3, 3, im::Bwmorph3Op::kRemove, vol_out.as_mut_ref()) ==
          im::ImageStatus::kOk);
    CHECK(vol_out[1 * 9 + 1 * 3 + 1] == 0 && vol_out[0] == 1);

    keel::Array<im::BoundaryPoint> pts;
    CHECK(im::bwtraceboundary_2d(filled.as_ref(), 7, 7, 1, 1, 2, 8, 0, true, pts) ==
          im::ImageStatus::kOk);
    CHECK(pts.size() >= 16 && pts[0].row == 1 && pts[0].col == 1);
}

void test_filter() {
    // 1. fspecial kernels
    im::Kernel2D k;
    CHECK(im::fspecial_average(3, 3, k) == im::ImageStatus::kOk);
    CHECK(k.rows == 3 && k.cols == 3);
    CHECK_NEAR(k.data_col_major[0], 1.0 / 9.0, 1e-12);

    CHECK(im::fspecial_gaussian(5, 5, 1.0, k) == im::ImageStatus::kOk);
    f64 gsum = 0.0;
    for (usize i = 0; i < k.data_col_major.size(); ++i) gsum += k.data_col_major[i];
    CHECK_NEAR(gsum, 1.0, 1e-12);

    CHECK(im::fspecial_disk(3.0, k) == im::ImageStatus::kOk);
    f64 dsum = 0.0;
    for (usize i = 0; i < k.data_col_major.size(); ++i) dsum += k.data_col_major[i];
    CHECK_NEAR(dsum, 1.0, 1e-12);

    CHECK(im::fspecial_motion(9.0, 30.0, k) == im::ImageStatus::kOk);
    f64 msum = 0.0;
    for (usize i = 0; i < k.data_col_major.size(); ++i) msum += k.data_col_major[i];
    CHECK_NEAR(msum, 1.0, 1e-10);

    // 2. imfilter_2d, medfilt2, imsharpen, imbilatfilt, stdfilt, wiener2
    keel::Array<f64> img(16 * 16, 10.0);
    img[8 * 16 + 8] = 100.0;
    keel::Array<f64> out(16 * 16, 0.0);
    CHECK(im::medfilt2_f64(img.as_ref(), 16, 16, 3, 3, out.as_mut_ref()) == im::ImageStatus::kOk);
    CHECK_NEAR(out[8 * 16 + 8], 10.0, 1e-12);

    keel::Array<u8> u_in(16 * 16, u8{10}), u_out(16 * 16, u8{0});
    u_in[8 * 16 + 8] = 250;
    CHECK(im::medfilt2_u8(u_in.as_ref(), 16, 16, 3, 3, u_out.as_mut_ref()) == im::ImageStatus::kOk);
    CHECK(u_out[8 * 16 + 8] == 10);
    CHECK(im::box_filter_2d_u8_replicate(u_in.as_ref(), 16, 16, 3, u_out.as_mut_ref()) ==
          im::ImageStatus::kOk);
    CHECK(u_out[8 * 16 + 8] == 37);

    CHECK(im::imsharpen_2d(img.as_ref(), 16, 16, 1.0, 0.8, 0.0, out.as_mut_ref()) ==
          im::ImageStatus::kOk);
    CHECK(out[8 * 16 + 8] > 100.0);

    CHECK(im::imbilatfilt_2d(img.as_ref(), 16, 16, 100.0, 1.0, out.as_mut_ref()) ==
          im::ImageStatus::kOk);
    CHECK(out[8 * 16 + 8] > 50.0);

    keel::Array<u8> dom(9, u8{1});
    CHECK(im::stdfilt_2d(img.as_ref(), 16, 16, dom.as_ref(), 3, 3, out.as_mut_ref()) ==
          im::ImageStatus::kOk);
    CHECK_NEAR(out[2 * 16 + 2], 0.0, 1e-12);
    CHECK(out[8 * 16 + 8] > 20.0);

    f64 w_noise = 0.0;
    CHECK(im::wiener2_2d(img.as_ref(), 16, 16, 3, 3, keel::Limits<f64>::quiet_nan(),
                         out.as_mut_ref(), w_noise) == im::ImageStatus::kOk);
    CHECK(w_noise >= 0.0);

    // 3. imdiffusefilt, imgaborfilt, imnlmfilt
    const f64 kt[3] = {5.0, 5.0, 5.0};
    CHECK(im::imdiffusefilt_2d(img.as_ref(), 16, 16, kt, true, true, out.as_mut_ref()) ==
          im::ImageStatus::kOk);
    CHECK(out[8 * 16 + 8] <= 100.0);

    keel::Array<f64> mag(16 * 16, 0.0), ph(16 * 16, 0.0);
    CHECK(im::imgaborfilt_2d(img.as_ref(), 16, 16, 4.0, 0.0, 1.0, 0.5, mag.as_mut_ref(),
                             ph.as_mut_ref()) == im::ImageStatus::kOk);
    CHECK(mag[8 * 16 + 8] > 0.0);

    keel::Array<f64> big(24 * 24, 5.0), nlm_out(24 * 24, 0.0);
    f64 est_dos = 0.0;
    CHECK(im::imnlmfilt_2d(big.as_ref(), 24, 24, -1.0, 5, 3, nlm_out.as_mut_ref(), est_dos) ==
          im::ImageStatus::kOk);
    CHECK_NEAR(nlm_out[12 * 24 + 12], 5.0, 1e-12);
    CHECK(est_dos > 0.0);
}

void test_transform() {
    // 1. dct_matrix orthogonality + dct_2d/idct_2d round-trip
    keel::Array<f64> d4(16, 0.0);
    CHECK(im::dct_matrix(4, d4.as_mut_ref()) == im::ImageStatus::kOk);
    // D * D^T == I
    for (usize r = 0; r < 4; ++r) {
        for (usize c = 0; c < 4; ++c) {
            f64 dot = 0.0;
            for (usize k = 0; k < 4; ++k) {
                dot += d4[k * 4 + r] * d4[k * 4 + c];
            }
            CHECK_NEAR(dot, (r == c) ? 1.0 : 0.0, 1e-12);
        }
    }

    keel::Array<f64> mat(4 * 5, 0.0), fwd(4 * 5, 0.0), inv(4 * 5, 0.0);
    for (usize i = 0; i < 20; ++i) mat[i] = keel::as<f64>(i + 1) * 0.5;
    CHECK(im::dct_2d(mat.as_ref(), 4, 5, fwd.as_mut_ref()) == im::ImageStatus::kOk);
    CHECK(im::idct_2d(fwd.as_ref(), 4, 5, inv.as_mut_ref()) == im::ImageStatus::kOk);
    for (usize i = 0; i < 20; ++i) {
        CHECK_NEAR(inv[i], mat[i], 1e-12);
    }

    // 2. integral_image_2d & integral_image_3d
    const f64 ones2d[6] = {1, 2, 3, 4, 5, 6}; // 2x3 col-major
    keel::Array<f64> sat2(3 * 4, 0.0);
    CHECK(im::integral_image_2d(ones2d, 2, 3, sat2.as_mut_ref()) == im::ImageStatus::kOk);
    CHECK_NEAR(sat2[3 * 3 + 2], 21.0, 1e-12);

    keel::Array<f64> ones3d(2 * 2 * 2, 1.0), sat3(3 * 3 * 3, 0.0);
    CHECK(im::integral_image_3d(ones3d.as_ref(), 2, 2, 2, sat3.as_mut_ref()) ==
          im::ImageStatus::kOk);
    CHECK_NEAR(sat3[2 * 9 + 2 * 3 + 2], 8.0, 1e-12);

    // 3. phantom & checkerboard
    keel::Array<f64> ell(60, 0.0);
    CHECK(im::phantom_ellipses(im::PhantomModel::kModifiedSheppLogan, ell.as_mut_ref()) ==
          im::ImageStatus::kOk);
    keel::Array<f64> ph(32 * 32, 0.0);
    CHECK(im::phantom_from_ellipses(ell.as_ref(), 10, 32, ph.as_mut_ref()) == im::ImageStatus::kOk);
    CHECK(ph[16 * 32 + 16] > 0.0);

    keel::Array<f64> cb(16 * 16, 0.0);
    CHECK(im::checkerboard(2, 4, 4, cb.as_mut_ref()) == im::ImageStatus::kOk);
    CHECK_NEAR(cb[0], 0.0, 1e-12);
    CHECK_NEAR(cb[2 * 16 + 0], 1.0, 1e-12);
    CHECK_NEAR(cb[10 * 16 + 0], 0.7, 1e-12);

    // 4. psf_to_otf / otf_to_psf round-trip
    const f64 psf[9] = {0.0, 0.1, 0.0, 0.1, 0.6, 0.1, 0.0, 0.1, 0.0};
    keel::Array<keel::c64> otf(8 * 8, keel::c64{0.0, 0.0});
    CHECK(im::psf_to_otf(psf, 3, 3, 8, 8, otf.as_mut_ref()) == im::ImageStatus::kOk);
    keel::Array<keel::c64> psf_back(3 * 3, keel::c64{0.0, 0.0});
    CHECK(im::otf_to_psf(otf.as_ref(), 8, 8, 3, 3, psf_back.as_mut_ref()) == im::ImageStatus::kOk);
    for (usize i = 0; i < 9; ++i) {
        CHECK_NEAR(psf_back[i].re, psf[i], 1e-12);
        CHECK_NEAR(psf_back[i].im, 0.0, 1e-12);
    }

    // 5. normxcorr2 exact peak at embedded template location
    keel::Array<f64> tmpl(3 * 3, 0.0);
    for (usize i = 0; i < 9; ++i) tmpl[i] = keel::as<f64>((i * 7 + 3) % 11);
    keel::Array<f64> big(8 * 8, 1.0);
    for (usize c = 0; c < 3; ++c) {
        for (usize r = 0; r < 3; ++r) {
            big[(c + 2) * 8 + (r + 3)] = tmpl[c * 3 + r];
        }
    }
    keel::Array<f64> ncc(10 * 10, 0.0);
    CHECK(im::normxcorr2(tmpl.as_ref(), 3, 3, big.as_ref(), 8, 8, ncc.as_mut_ref()) ==
          im::ImageStatus::kOk);
    CHECK_NEAR(ncc[(2 + 2) * 10 + (3 + 2)], 1.0, 1e-11);

    // 6. deconvwnr_2d, edgetaper_2d, deconvreg_2d
    keel::Array<f64> img(16 * 16, 0.0), out(16 * 16, 0.0);
    for (usize c = 0; c < 16; ++c) {
        for (usize r = 0; r < 16; ++r) {
            img[c * 16 + r] = ((r >= 4 && r <= 11 && c >= 4 && c <= 11) ? 1.0 : 0.1);
        }
    }
    const f64 nsr[1] = {0.01};
    const f64 sx1[1] = {1.0};
    CHECK(im::deconvwnr_2d(img.as_ref(), 16, 16, psf, 3, 3, nsr, sx1, out.as_mut_ref()) ==
          im::ImageStatus::kOk);
    CHECK(out[8 * 16 + 8] > 0.5);

    CHECK(im::edgetaper_2d(img.as_ref(), 16, 16, psf, 3, 3, out.as_mut_ref()) ==
          im::ImageStatus::kOk);
    CHECK_NEAR(out[8 * 16 + 8], 1.0, 1e-10);

    f64 lam_out = 0.0;
    CHECK(im::deconvreg_2d(img.as_ref(), 16, 16, psf, 3, 3, 0.05, 1e-6, 1.0,
                           keel::ArrayRef<f64>{}, 0, 0, out.as_mut_ref(), lam_out) ==
          im::ImageStatus::kOk);
    CHECK(lam_out >= 1e-6 && lam_out <= 1.0);
    CHECK(out[8 * 16 + 8] > 0.5);
}

void test_region() {
    // 6x6 image with a 3x3 square (rows 1..3, cols 1..3) and a 1x1 pixel at (5,5)
    keel::Array<u8> bw(36, u8{0});
    for (usize c = 1; c <= 3; ++c) {
        for (usize r = 1; r <= 3; ++r) {
            bw[c * 6 + r] = 1;
        }
    }
    bw[5 * 6 + 5] = 1;

    keel::Array<i32> labels(36, 0);
    i32 num_k = 0;
    CHECK(im::bwlabel_2d(bw.as_ref(), 6, 6, 8, labels.as_mut_ref(), num_k) == im::ImageStatus::kOk);
    CHECK(num_k == 2);

    keel::Array<u8> opened(36, u8{0});
    CHECK(im::bwareaopen_2d(bw.as_ref(), 6, 6, 2, 8, opened.as_mut_ref()) == im::ImageStatus::kOk);
    CHECK(opened[2 * 6 + 2] == 1);
    CHECK(opened[5 * 6 + 5] == 0);

    keel::Array<u8> perim(36, u8{0});
    CHECK(im::bwperim_2d(bw.as_ref(), 6, 6, 4, perim.as_mut_ref()) == im::ImageStatus::kOk);
    CHECK(perim[2 * 6 + 2] == 0);
    CHECK(perim[1 * 6 + 1] == 1);

    f64 euler = 0.0;
    CHECK(im::bweuler_2d(bw.as_ref(), 6, 6, 8, euler) == im::ImageStatus::kOk);
    CHECK_NEAR(euler, 2.0, 1e-12);

    keel::Array<f64> dist(36, 0.0);
    CHECK(im::bwdist_euclidean_2d(bw.as_ref(), 6, 6, dist.as_mut_ref()) == im::ImageStatus::kOk);
    CHECK_NEAR(dist[0], keel::sqrt(2.0), 1e-12);
    CHECK_NEAR(dist[1 * 6 + 1], 0.0, 1e-12);

    keel::Array<im::BoundaryPoint> bpts;
    keel::Array<usize> boff;
    CHECK(im::bwboundaries_2d(bw.as_ref(), 6, 6, 8, labels.as_mut_ref(), num_k, bpts, boff) ==
          im::ImageStatus::kOk);
    CHECK(num_k == 2);
    CHECK(boff.size() == 3);
    CHECK(boff[1] - boff[0] == 9); // 8 boundary pixels + closing point
    CHECK(boff[2] - boff[1] == 1); // single-pixel component

    keel::Array<im::RegionProps2D> props;
    CHECK(im::region_props_2d(labels.as_ref(), 6, 6, num_k,
                              im::kRegionPropEllipse | im::kRegionPropPerimeter, props) ==
          im::ImageStatus::kOk);
    CHECK(props.size() == 2);
    CHECK(props[0].area == 9);
    CHECK_NEAR(props[0].centroid_x, 2.0, 1e-12);
    CHECK_NEAR(props[0].centroid_y, 2.0, 1e-12);
    CHECK_NEAR(props[0].extent, 1.0, 1e-12);
    CHECK(props[0].perimeter > 7.0);

    // Component attributes (ConvexArea, Solidity, FilledArea)
    const i32 prs[9] = {1, 2, 3, 1, 2, 3, 1, 2, 3};
    const i32 pcs[9] = {1, 1, 1, 2, 2, 2, 3, 3, 3};
    const usize offs[2] = {0, 9};
    f64 attr_out[1] = {0.0};
    CHECK(im::region_component_attrs(prs, pcs, offs, 6, 6, 8, im::RegionAttr::kConvexArea,
                                     keel::ArrayRef<f64>{}, attr_out) == im::ImageStatus::kOk);
    CHECK_NEAR(attr_out[0], 9.0, 1e-12);
    CHECK(im::region_component_attrs(prs, pcs, offs, 6, 6, 8, im::RegionAttr::kSolidity,
                                     keel::ArrayRef<f64>{}, attr_out) == im::ImageStatus::kOk);
    CHECK_NEAR(attr_out[0], 1.0, 1e-12);

    // Freeman chain code on a 2x2 closed loop
    const f64 fr[4] = {1.0, 1.0, 2.0, 2.0};
    const f64 fc[4] = {1.0, 2.0, 2.0, 1.0};
    keel::Array<f64> fcc, fdiff;
    CHECK(im::freeman_chain_code(fr, fc, fcc, fdiff) == im::ImageStatus::kOk);
    CHECK(fcc.size() == 4);
    CHECK_NEAR(fcc[0], 0.0, 1e-12);
    CHECK_NEAR(fcc[1], 6.0, 1e-12);
}

void test_features_and_segment() {
    // 1. 2D gradient + edge (Sobel & Canny) + cornermetric + corner_peaks
    keel::Array<f64> img(16 * 16, 0.0);
    for (usize c = 4; c <= 11; ++c) {
        for (usize r = 4; r <= 11; ++r) {
            img[c * 16 + r] = 1.0;
        }
    }
    keel::Array<f64> gx(16 * 16, 0.0), gy(16 * 16, 0.0);
    CHECK(im::imgradientxy_2d(img.as_ref(), 16, 16, im::GradientMethod::kSobel, gx.as_mut_ref(),
                              gy.as_mut_ref()) == im::ImageStatus::kOk);
    CHECK(gx[4 * 16 + 8] > 0.0);
    CHECK(gy[8 * 16 + 4] > 0.0);

    keel::Array<u8> edges(16 * 16, u8{0});
    CHECK(im::edge_2d(img.as_ref(), 16, 16, im::EdgeMethod::kCanny, keel::kNaN,
                      edges.as_mut_ref()) == im::ImageStatus::kOk);
    usize edge_cnt = 0;
    for (usize i = 0; i < edges.size(); ++i) edge_cnt += edges[i];
    CHECK(edge_cnt > 10);

    keel::Array<f64> cm(16 * 16, 0.0);
    CHECK(im::cornermetric_2d(img.as_ref(), 16, 16, im::CornerMethod::kHarris, 0.04,
                              keel::ArrayRef<f64>{}, cm.as_mut_ref()) == im::ImageStatus::kOk);
    keel::Array<im::BoundaryPoint> cpeaks;
    CHECK(im::corner_peaks_2d(cm.as_ref(), 16, 16, 10, 0.01, cpeaks) == im::ImageStatus::kOk);
    CHECK(cpeaks.size() >= 4);

    // 2. Hough line detection on a horizontal line
    keel::Array<u8> line_bw(20 * 20, u8{0});
    for (usize c = 2; c <= 17; ++c) line_bw[c * 20 + 10] = 1;
    keel::Array<f64> thetas;
    for (i32 t = -90; t <= 89; ++t) thetas.push_back(keel::as<f64>(t));
    keel::Array<f64> h_acc, rho;
    CHECK(im::hough_2d(line_bw.as_ref(), 20, 20, 1.0, thetas.as_ref(), h_acc, rho) ==
          im::ImageStatus::kOk);
    keel::Array<im::BoundaryPoint> hpeaks;
    CHECK(im::houghpeaks_2d(h_acc.as_ref(), rho.size(), thetas.size(), 1, -1.0, 0, 0,
                            thetas.as_ref(), hpeaks) == im::ImageStatus::kOk);
    CHECK(hpeaks.size() == 1);
    keel::Array<im::HoughLine> hlines;
    CHECK(im::houghlines_2d(line_bw.as_ref(), 20, 20, thetas.as_ref(), rho.as_ref(),
                            hpeaks.as_ref(), 5.0, 5.0, hlines) == im::ImageStatus::kOk);
    CHECK(hlines.size() == 1);

    // 3. imfindcircles_2d on a Gaussian-smoothed bright disk of radius 15 centered at (29, 24)
    keel::Array<f64> circ_raw(60 * 60, 0.0), circ(60 * 60, 0.0);
    for (usize c = 0; c < 60; ++c) {
        for (usize r = 0; r < 60; ++r) {
            const f64 dx = keel::as<f64>(c) - 29.0;
            const f64 dy = keel::as<f64>(r) - 24.0;
            if (dx * dx + dy * dy <= 225.0) circ_raw[c * 60 + r] = 1.0;
        }
    }
    im::Kernel2D gk;
    CHECK(im::fspecial_gaussian(5, 5, 1.5, gk) == im::ImageStatus::kOk);
    CHECK(im::imfilter_2d(circ_raw.as_ref(), 60, 60, gk.data_col_major.as_ref(), 5, 5,
                          im::PadMode::kReplicate, 0.0, false, false, circ.as_mut_ref()) ==
          im::ImageStatus::kOk);
    keel::Array<im::CircleDetection> circles;
    CHECK(im::imfindcircles_2d(circ.as_ref(), 60, 60, 10.0, 20.0, im::CirclePolarity::kBright,
                               im::CircleMethod::kPhaseCode, 0.85, -1.0, false, circles) ==
          im::ImageStatus::kOk);
    CHECK(circles.size() == 1);
    if (!circles.empty()) {
        CHECK_NEAR(circles[0].center_x, 29.0, 0.5);
        CHECK_NEAR(circles[0].center_y, 24.0, 0.5);
        CHECK_NEAR(circles[0].radius, 15.0, 0.5);
    }

    // 4. Segmentation: dice/jaccard, regionfill, poly2mask, graydist, reducepoly, watershed
    const u8 ma[4] = {1, 1, 0, 0};
    const u8 mb[4] = {1, 0, 1, 0};
    CHECK_NEAR(im::dice_coeff(ma, mb), 0.5, 1e-12);
    CHECK_NEAR(im::jaccard_index(ma, mb), 1.0 / 3.0, 1e-12);

    keel::Array<f64> hole_img(5 * 5, 10.0), hole_out(5 * 5, 0.0);
    hole_img[2 * 5 + 2] = 0.0;
    keel::Array<u8> hole_mask(5 * 5, u8{0});
    hole_mask[2 * 5 + 2] = 1;
    CHECK(im::regionfill_2d(hole_img.as_ref(), hole_mask.as_ref(), 5, 5, hole_out.as_mut_ref()) ==
          im::ImageStatus::kOk);
    CHECK_NEAR(hole_out[2 * 5 + 2], 10.0, 1e-10);

    const f64 px[4] = {2.0, 5.0, 5.0, 2.0};
    const f64 py[4] = {2.0, 2.0, 5.0, 5.0};
    keel::Array<u8> pmask(6 * 6, u8{0});
    CHECK(im::poly2mask_2d(px, py, 6, 6, pmask.as_mut_ref()) == im::ImageStatus::kOk);
    CHECK(pmask[3 * 6 + 3] == 1);
    CHECK(pmask[0] == 0);

    keel::Array<f64> gimg(5 * 5, 1.0), gdist(5 * 5, 0.0);
    const usize seeds[1] = {0};
    CHECK(im::graydist_2d(gimg.as_ref(), 5, 5, seeds, im::GeodesicMetric::kCityblock,
                          gdist.as_mut_ref()) == im::ImageStatus::kOk);
    CHECK_NEAR(gdist[4 * 5 + 4], 8.0, 1e-12);

    const f64 rpx[5] = {0.0, 1.0, 2.0, 3.0, 4.0};
    const f64 rpy[5] = {0.0, 0.01, 0.0, -0.01, 0.0};
    keel::Array<i32> kept;
    CHECK(im::reducepoly_f64(rpx, rpy, 0.05, kept) == im::ImageStatus::kOk);
    CHECK(kept.size() == 2);

    keel::Array<f64> ws_in(7 * 7, 10.0);
    ws_in[1 * 7 + 3] = 0.0;
    ws_in[5 * 7 + 3] = 0.0;
    keel::Array<u32> ws_lab(7 * 7, u32{0});
    u32 nbasins = 0;
    CHECK(im::watershed_2d(ws_in.as_ref(), 7, 7, 8, ws_lab.as_mut_ref(), nbasins) ==
          im::ImageStatus::kOk);
    CHECK(nbasins == 2);

    // 5. Wide-RGB BT.2020 / BT.2100 + palette helpers
    const u16 wr[2] = {64, 940};
    const u16 wg[2] = {64, 500};
    const u16 wb[2] = {64, 200};
    u16 wy[2] = {}, wcb[2] = {}, wcr[2] = {}, rr[2] = {}, rg[2] = {}, rb[2] = {};
    CHECK(im::rgbwide_to_ycbcr_planar(wr, wg, wb, 10, wy, wcb, wcr) == im::ImageStatus::kOk);
    CHECK(im::ycbcr_to_rgbwide_planar(wy, wcb, wcr, 10, rr, rg, rb) == im::ImageStatus::kOk);
    CHECK(rr[0] == 64 && rg[0] == 64 && rb[0] == 64);
    CHECK(keel::abs(keel::as<i32>(rr[1]) - 940) <= 2);

    const f64 wrd[2] = {64.0, 940.0};
    const f64 wgd[2] = {64.0, 500.0};
    const f64 wbd[2] = {64.0, 200.0};
    f64 wx[2] = {}, wy2[2] = {}, wz[2] = {};
    CHECK(im::rgbwide_to_xyz_planar(wrd, wgd, wbd, 10, im::WideColorSpace::kBt2020, wx, wy2, wz) ==
          im::ImageStatus::kOk);
    CHECK(im::xyz_to_rgbwide_planar(wx, wy2, wz, 10, im::WideColorSpace::kBt2020, rr, rg, rb) ==
          im::ImageStatus::kOk);
    CHECK(rr[0] == 64 && rr[1] == 940 && rg[1] == 500 && rb[1] == 200);

    const f64 cmr[3] = {1.0, 0.0, 1.0};
    const f64 cmg[3] = {0.0, 1.0, 0.0};
    const f64 cmb[3] = {0.0, 0.0, 0.0};
    const usize idx_in[3] = {0, 1, 2};
    usize idx_out[3] = {};
    keel::Array<f64> ur, ug, ub;
    CHECK(im::colormap_unique(idx_in, cmr, cmg, cmb, idx_out, ur, ug, ub) == im::ImageStatus::kOk);
    CHECK(ur.size() == 2);
    CHECK(idx_out[0] == idx_out[2]);
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
    test_morph();
    test_filter();
    test_transform();
    test_region();
    test_features_and_segment();

    if (g_failures == 0) {
        std::printf("ALL keel::image TESTS PASSED\n");
        return 0;
    }
    std::fprintf(stderr, "TOTAL FAILURES: %d\n", g_failures);
    return 1;
}
