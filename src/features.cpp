#include "keel/image.h"

#include "keel/algo.h"
#include "keel/math.h"
#include "keel/static_array.h"

namespace keel::image {

namespace {

void conv3d_replicate(ArrayRef<f64> vol, usize height, usize width, usize depth, const f64 k[27],
                      MutArrayRef<f64> out) noexcept {
    auto clamp_idx = [](i64 x, usize n) noexcept -> usize {
        if (x < 0) return 0;
        if (x >= as<i64>(n)) return n - 1;
        return as<usize>(x);
    };
    const usize plane = height * width;
    for (usize p = 0; p < depth; ++p) {
        for (usize c = 0; c < width; ++c) {
            for (usize r = 0; r < height; ++r) {
                f64 sum = 0.0;
                for (i32 dp = -1; dp <= 1; ++dp) {
                    const usize pp = clamp_idx(as<i64>(p) + dp, depth);
                    for (i32 dc = -1; dc <= 1; ++dc) {
                        const usize cc = clamp_idx(as<i64>(c) + dc, width);
                        for (i32 dr = -1; dr <= 1; ++dr) {
                            const usize rr = clamp_idx(as<i64>(r) + dr, height);
                            const i32 ki = (dr + 1) + 3 * (dc + 1) + 9 * (dp + 1);
                            sum += k[ki] * vol[rr + height * cc + plane * pp];
                        }
                    }
                }
                out[r + height * c + plane * p] = sum;
            }
        }
    }
}

void build_3x3x3_kernel(const f64 rows3[3][3][3], f64 k[27]) noexcept {
    for (i32 p = 0; p < 3; ++p) {
        for (i32 c = 0; c < 3; ++c) {
            for (i32 r = 0; r < 3; ++r) {
                k[r + 3 * c + 9 * p] = rows3[p][r][c];
            }
        }
    }
}

f64 max_abs(ArrayRef<f64> v) noexcept {
    f64 mx = 0.0;
    for (usize i = 0; i < v.size(); ++i) {
        const f64 a = abs(v[i]);
        if (a > mx) {
            mx = a;
        }
    }
    return mx;
}

ImageStatus canny_edge_2d(ArrayRef<f64> img, usize height, usize width, f64 high_frac_in,
                          MutArrayRef<u8> out, Allocator* a) {
    const usize total = height * width;
    fill(out.begin(), out.end(), u8{0});
    if (total == 0) {
        return ImageStatus::kOk;
    }

    const f64 sigma = sqrt(2.0);
    const i32 hw = max(i32{1}, as<i32>(ceil(3.0 * sigma)));
    const usize l = as<usize>(2 * hw + 1);
    Array<f64> g(a);
    g.resize(l);
    Array<f64> dg(a);
    dg.resize(l);
    f64 gsum = 0.0;
    for (usize i = 0; i < l; ++i) {
        const f64 x = as<f64>(as<i32>(i) - hw);
        g[i] = exp(-(x * x) / (2.0 * sigma * sigma));
        gsum += g[i];
    }
    for (usize i = 0; i < l; ++i) {
        g[i] /= gsum;
        const f64 x = as<f64>(as<i32>(i) - hw);
        dg[i] = -x * exp(-(x * x) / (2.0 * sigma * sigma));
    }

    Array<f64> tmp(a);
    tmp.resize(total);
    Array<f64> gx(a);
    gx.resize(total);
    Array<f64> gy(a);
    gy.resize(total);

    (void)imfilter_2d(img, height, width, g.as_ref(), l, 1, PadMode::kReplicate, 0.0, false, false,
                      tmp.as_mut_ref());
    (void)imfilter_2d(tmp.as_ref(), height, width, dg.as_ref(), 1, l, PadMode::kReplicate, 0.0,
                      false, false, gx.as_mut_ref());
    (void)imfilter_2d(img, height, width, g.as_ref(), 1, l, PadMode::kReplicate, 0.0, false, false,
                      tmp.as_mut_ref());
    (void)imfilter_2d(tmp.as_ref(), height, width, dg.as_ref(), l, 1, PadMode::kReplicate, 0.0,
                      false, false, gy.as_mut_ref());

    Array<f64> mag(a);
    mag.resize(total);
    f64 max_mag = 0.0;
    for (usize i = 0; i < total; ++i) {
        const f64 m = sqrt(gx[i] * gx[i] + gy[i] * gy[i]);
        mag[i] = m;
        if (m > max_mag) {
            max_mag = m;
        }
    }
    if (max_mag <= 0.0) {
        return ImageStatus::kOk;
    }

    auto at = [&](usize r, usize c) noexcept -> f64 { return mag[c * height + r]; };

    Array<f64> nms(a);
    nms.assign(total, 0.0);
    for (usize c = 1; c + 1 < width; ++c) {
        for (usize r = 1; r + 1 < height; ++r) {
            const usize i = c * height + r;
            const f64 m = mag[i];
            if (m <= 0.0) {
                continue;
            }
            const f64 gxv = gx[i];
            const f64 gyv = gy[i];
            const f64 agx = abs(gxv);
            const f64 agy = abs(gyv);
            f64 mf = 0.0;
            f64 mb = 0.0;
            if (agx >= agy) {
                const f64 w = (agx > 0.0) ? (agy / agx) : 0.0;
                if (gxv * gyv >= 0.0) {
                    mf = (1.0 - w) * at(r, c + 1) + w * at(r + 1, c + 1);
                    mb = (1.0 - w) * at(r, c - 1) + w * at(r - 1, c - 1);
                } else {
                    mf = (1.0 - w) * at(r, c + 1) + w * at(r - 1, c + 1);
                    mb = (1.0 - w) * at(r, c - 1) + w * at(r + 1, c - 1);
                }
            } else {
                const f64 w = (agy > 0.0) ? (agx / agy) : 0.0;
                if (gxv * gyv >= 0.0) {
                    mf = (1.0 - w) * at(r + 1, c) + w * at(r + 1, c + 1);
                    mb = (1.0 - w) * at(r - 1, c) + w * at(r - 1, c - 1);
                } else {
                    mf = (1.0 - w) * at(r + 1, c) + w * at(r + 1, c - 1);
                    mb = (1.0 - w) * at(r - 1, c) + w * at(r - 1, c + 1);
                }
            }
            if (m >= mf && m >= mb) {
                nms[i] = m;
            }
        }
    }

    const f64 high_frac = (is_nan(high_frac_in) || high_frac_in <= 0.0) ? 0.2 : high_frac_in;
    const f64 high_t = high_frac * max_mag;
    const f64 low_t = 0.4 * high_t;

    Array<usize> stack(a);
    stack.reserve(total / 8 + 1);
    for (usize i = 0; i < total; ++i) {
        if (nms[i] >= high_t) {
            out[i] = 1;
            stack.push_back(i);
        }
    }
    static constexpr i32 kDr8[8] = {-1, -1, -1, 0, 0, 1, 1, 1};
    static constexpr i32 kDc8[8] = {-1, 0, 1, -1, 1, -1, 0, 1};
    const i32 h = as<i32>(height);
    const i32 w = as<i32>(width);
    while (!stack.empty()) {
        const usize p = stack.back();
        stack.pop_back();
        const i32 r = as<i32>(p % height);
        const i32 c = as<i32>(p / height);
        for (i32 k = 0; k < 8; ++k) {
            const i32 nr = r + kDr8[k];
            const i32 nc = c + kDc8[k];
            if (nr < 0 || nr >= h || nc < 0 || nc >= w) {
                continue;
            }
            const usize q = as<usize>(nc) * height + as<usize>(nr);
            if (!out[q] && nms[q] >= low_t) {
                out[q] = 1;
                stack.push_back(q);
            }
        }
    }
    return ImageStatus::kOk;
}

} // namespace

ImageStatus imgradientxy_2d(ArrayRef<f64> img_col_major, usize height, usize width,
                            GradientMethod method, MutArrayRef<f64> out_gx_col_major,
                            MutArrayRef<f64> out_gy_col_major) noexcept {
    const usize total = height * width;
    if (img_col_major.size() != total || out_gx_col_major.size() != total
        || out_gy_col_major.size() != total) {
        return ImageStatus::kInvalidArgument;
    }
    if (total == 0) {
        return ImageStatus::kOk;
    }
    // Kernels stored column-major for imfilter_2d (correlation, replicate border).
    switch (method) {
        case GradientMethod::kPrewitt: {
            static constexpr f64 kKx[9] = {-1, -1, -1, 0, 0, 0, 1, 1, 1};
            static constexpr f64 kKy[9] = {-1, 0, 1, -1, 0, 1, -1, 0, 1};
            (void)imfilter_2d(img_col_major, height, width, ArrayRef<f64>(kKx, 9), 3, 3,
                              PadMode::kReplicate, 0.0, false, false, out_gx_col_major);
            (void)imfilter_2d(img_col_major, height, width, ArrayRef<f64>(kKy, 9), 3, 3,
                              PadMode::kReplicate, 0.0, false, false, out_gy_col_major);
            break;
        }
        case GradientMethod::kCentral: {
            for (usize r = 0; r < height; ++r) {
                for (usize c = 0; c < width; ++c) {
                    const usize idx = c * height + r;
                    if (width == 1) {
                        out_gx_col_major[idx] = 0.0;
                    } else if (c == 0) {
                        out_gx_col_major[idx] =
                            img_col_major[1 * height + r] - img_col_major[0 * height + r];
                    } else if (c == width - 1) {
                        out_gx_col_major[idx] = img_col_major[(width - 1) * height + r] -
                                                img_col_major[(width - 2) * height + r];
                    } else {
                        out_gx_col_major[idx] =
                            0.5 * (img_col_major[(c + 1) * height + r] -
                                   img_col_major[(c - 1) * height + r]);
                    }
                }
            }
            for (usize c = 0; c < width; ++c) {
                const usize col_off = c * height;
                for (usize r = 0; r < height; ++r) {
                    const usize idx = col_off + r;
                    if (height == 1) {
                        out_gy_col_major[idx] = 0.0;
                    } else if (r == 0) {
                        out_gy_col_major[idx] =
                            img_col_major[col_off + 1] - img_col_major[col_off + 0];
                    } else if (r == height - 1) {
                        out_gy_col_major[idx] = img_col_major[col_off + (height - 1)] -
                                                img_col_major[col_off + (height - 2)];
                    } else {
                        out_gy_col_major[idx] =
                            0.5 * (img_col_major[col_off + (r + 1)] -
                                   img_col_major[col_off + (r - 1)]);
                    }
                }
            }
            break;
        }
        case GradientMethod::kIntermediate: {
            static constexpr f64 kK[2] = {-1.0, 1.0};
            (void)imfilter_2d(img_col_major, height, width, ArrayRef<f64>(kK, 2), 1, 2,
                              PadMode::kReplicate, 0.0, false, false, out_gx_col_major);
            (void)imfilter_2d(img_col_major, height, width, ArrayRef<f64>(kK, 2), 2, 1,
                              PadMode::kReplicate, 0.0, false, false, out_gy_col_major);
            break;
        }
        case GradientMethod::kSobel:
        default: {
            static constexpr f64 kKx[9] = {-1, -2, -1, 0, 0, 0, 1, 2, 1};
            static constexpr f64 kKy[9] = {-1, 0, 1, -2, 0, 2, -1, 0, 1};
            (void)imfilter_2d(img_col_major, height, width, ArrayRef<f64>(kKx, 9), 3, 3,
                              PadMode::kReplicate, 0.0, false, false, out_gx_col_major);
            (void)imfilter_2d(img_col_major, height, width, ArrayRef<f64>(kKy, 9), 3, 3,
                              PadMode::kReplicate, 0.0, false, false, out_gy_col_major);
            break;
        }
    }
    return ImageStatus::kOk;
}

ImageStatus imgradientxyz_3d(ArrayRef<f64> vol_col_major, usize height, usize width, usize depth,
                             GradientMethod method, MutArrayRef<f64> out_gx,
                             MutArrayRef<f64> out_gy, MutArrayRef<f64> out_gz) noexcept {
    const usize total = height * width * depth;
    if (vol_col_major.size() != total || out_gx.size() != total || out_gy.size() != total
        || out_gz.size() != total) {
        return ImageStatus::kInvalidArgument;
    }
    if (total == 0) {
        return ImageStatus::kOk;
    }
    if (method == GradientMethod::kSobel) {
        static constexpr f64 kHx[3][3][3] = {
            {{-1, 0, 1}, {-3, 0, 3}, {-1, 0, 1}},
            {{-3, 0, 3}, {-6, 0, 6}, {-3, 0, 3}},
            {{-1, 0, 1}, {-3, 0, 3}, {-1, 0, 1}},
        };
        static constexpr f64 kHy[3][3][3] = {
            {{-1, -3, -1}, {0, 0, 0}, {1, 3, 1}},
            {{-3, -6, -3}, {0, 0, 0}, {3, 6, 3}},
            {{-1, -3, -1}, {0, 0, 0}, {1, 3, 1}},
        };
        static constexpr f64 kHz[3][3][3] = {
            {{-1, -3, -1}, {-3, -6, -3}, {-1, -3, -1}},
            {{0, 0, 0}, {0, 0, 0}, {0, 0, 0}},
            {{1, 3, 1}, {3, 6, 3}, {1, 3, 1}},
        };
        f64 kx[27];
        f64 ky[27];
        f64 kz[27];
        build_3x3x3_kernel(kHx, kx);
        build_3x3x3_kernel(kHy, ky);
        build_3x3x3_kernel(kHz, kz);
        conv3d_replicate(vol_col_major, height, width, depth, kx, out_gx);
        conv3d_replicate(vol_col_major, height, width, depth, ky, out_gy);
        conv3d_replicate(vol_col_major, height, width, depth, kz, out_gz);
        return ImageStatus::kOk;
    }
    if (method == GradientMethod::kPrewitt) {
        static constexpr f64 kHx[3][3][3] = {
            {{-1, 0, 1}, {-1, 0, 1}, {-1, 0, 1}},
            {{-1, 0, 1}, {-1, 0, 1}, {-1, 0, 1}},
            {{-1, 0, 1}, {-1, 0, 1}, {-1, 0, 1}},
        };
        static constexpr f64 kHy[3][3][3] = {
            {{-1, -1, -1}, {0, 0, 0}, {1, 1, 1}},
            {{-1, -1, -1}, {0, 0, 0}, {1, 1, 1}},
            {{-1, -1, -1}, {0, 0, 0}, {1, 1, 1}},
        };
        static constexpr f64 kHz[3][3][3] = {
            {{-1, -1, -1}, {-1, -1, -1}, {-1, -1, -1}},
            {{0, 0, 0}, {0, 0, 0}, {0, 0, 0}},
            {{1, 1, 1}, {1, 1, 1}, {1, 1, 1}},
        };
        f64 kx[27];
        f64 ky[27];
        f64 kz[27];
        build_3x3x3_kernel(kHx, kx);
        build_3x3x3_kernel(kHy, ky);
        build_3x3x3_kernel(kHz, kz);
        conv3d_replicate(vol_col_major, height, width, depth, kx, out_gx);
        conv3d_replicate(vol_col_major, height, width, depth, ky, out_gy);
        conv3d_replicate(vol_col_major, height, width, depth, kz, out_gz);
        return ImageStatus::kOk;
    }

    const usize plane = height * width;
    auto idx = [&](usize r, usize c, usize p) noexcept -> usize {
        return r + height * c + plane * p;
    };
    fill(out_gx.begin(), out_gx.end(), 0.0);
    fill(out_gy.begin(), out_gy.end(), 0.0);
    fill(out_gz.begin(), out_gz.end(), 0.0);

    if (method == GradientMethod::kCentral) {
        for (usize p = 0; p < depth; ++p) {
            for (usize r = 0; r < height; ++r) {
                for (usize c = 0; c < width; ++c) {
                    if (width == 1) {
                        out_gx[idx(r, c, p)] = 0.0;
                    } else if (c == 0) {
                        out_gx[idx(r, c, p)] = vol_col_major[idx(r, 1, p)] - vol_col_major[idx(r, 0, p)];
                    } else if (c == width - 1) {
                        out_gx[idx(r, c, p)] =
                            vol_col_major[idx(r, width - 1, p)] - vol_col_major[idx(r, width - 2, p)];
                    } else {
                        out_gx[idx(r, c, p)] =
                            0.5 * (vol_col_major[idx(r, c + 1, p)] - vol_col_major[idx(r, c - 1, p)]);
                    }
                }
            }
        }
        for (usize p = 0; p < depth; ++p) {
            for (usize c = 0; c < width; ++c) {
                for (usize r = 0; r < height; ++r) {
                    if (height == 1) {
                        out_gy[idx(r, c, p)] = 0.0;
                    } else if (r == 0) {
                        out_gy[idx(r, c, p)] = vol_col_major[idx(1, c, p)] - vol_col_major[idx(0, c, p)];
                    } else if (r == height - 1) {
                        out_gy[idx(r, c, p)] =
                            vol_col_major[idx(height - 1, c, p)] - vol_col_major[idx(height - 2, c, p)];
                    } else {
                        out_gy[idx(r, c, p)] =
                            0.5 * (vol_col_major[idx(r + 1, c, p)] - vol_col_major[idx(r - 1, c, p)]);
                    }
                }
            }
        }
        for (usize c = 0; c < width; ++c) {
            for (usize r = 0; r < height; ++r) {
                for (usize p = 0; p < depth; ++p) {
                    if (depth == 1) {
                        out_gz[idx(r, c, p)] = 0.0;
                    } else if (p == 0) {
                        out_gz[idx(r, c, p)] = vol_col_major[idx(r, c, 1)] - vol_col_major[idx(r, c, 0)];
                    } else if (p == depth - 1) {
                        out_gz[idx(r, c, p)] =
                            vol_col_major[idx(r, c, depth - 1)] - vol_col_major[idx(r, c, depth - 2)];
                    } else {
                        out_gz[idx(r, c, p)] =
                            0.5 * (vol_col_major[idx(r, c, p + 1)] - vol_col_major[idx(r, c, p - 1)]);
                    }
                }
            }
        }
        return ImageStatus::kOk;
    }

    // Intermediate: forward diff with trailing slice zero.
    for (usize p = 0; p < depth; ++p) {
        for (usize r = 0; r < height; ++r) {
            for (usize c = 0; c + 1 < width; ++c) {
                out_gx[idx(r, c, p)] = vol_col_major[idx(r, c + 1, p)] - vol_col_major[idx(r, c, p)];
            }
        }
    }
    for (usize p = 0; p < depth; ++p) {
        for (usize c = 0; c < width; ++c) {
            for (usize r = 0; r + 1 < height; ++r) {
                out_gy[idx(r, c, p)] = vol_col_major[idx(r + 1, c, p)] - vol_col_major[idx(r, c, p)];
            }
        }
    }
    for (usize c = 0; c < width; ++c) {
        for (usize r = 0; r < height; ++r) {
            for (usize p = 0; p + 1 < depth; ++p) {
                out_gz[idx(r, c, p)] = vol_col_major[idx(r, c, p + 1)] - vol_col_major[idx(r, c, p)];
            }
        }
    }
    return ImageStatus::kOk;
}

ImageStatus edge_2d(ArrayRef<f64> img_col_major, usize height, usize width, EdgeMethod method,
                    f64 thresh, MutArrayRef<u8> out_col_major, Allocator* alloc) {
    const usize total = height * width;
    if (img_col_major.size() != total || out_col_major.size() != total) {
        return ImageStatus::kInvalidArgument;
    }
    if (total == 0) {
        return ImageStatus::kOk;
    }
    Allocator* a = alloc ? alloc : default_allocator();
    if (method == EdgeMethod::kCanny) {
        return canny_edge_2d(img_col_major, height, width, thresh, out_col_major, a);
    }
    if (method == EdgeMethod::kLog || method == EdgeMethod::kZeroCross) {
        Kernel2D klog;
        const ImageStatus st = fspecial_log(5, 5, 0.5, klog, a);
        if (st != ImageStatus::kOk) {
            return st;
        }
        Array<f64> y(a);
        y.resize(total);
        (void)imfilter_2d(img_col_major, height, width, klog.data_col_major.as_ref(), klog.rows,
                          klog.cols, PadMode::kReplicate, 0.0, false, false, y.as_mut_ref());
        const f64 th = is_nan(thresh) ? (0.05 * max_abs(y.as_ref())) : thresh;
        const i32 h = as<i32>(height);
        const i32 w = as<i32>(width);
        for (i32 c = 0; c < w; ++c) {
            for (i32 r = 0; r < h; ++r) {
                const f64 av = y[as<usize>(c) * height + as<usize>(r)];
                bool zc = false;
                for (i32 dr = -1; dr <= 1 && !zc; ++dr) {
                    for (i32 dc = -1; dc <= 1 && !zc; ++dc) {
                        if (dr == 0 && dc == 0) {
                            continue;
                        }
                        const i32 rr = r + dr;
                        const i32 cc = c + dc;
                        if (rr < 0 || rr >= h || cc < 0 || cc >= w) {
                            continue;
                        }
                        const f64 bv = y[as<usize>(cc) * height + as<usize>(rr)];
                        if (av * bv < 0.0 && abs(av - bv) > th) {
                            zc = true;
                        }
                    }
                }
                out_col_major[as<usize>(c) * height + as<usize>(r)] = zc ? u8{1} : u8{0};
            }
        }
        return ImageStatus::kOk;
    }

    Array<f64> gx(a);
    gx.resize(total);
    Array<f64> gy(a);
    gy.resize(total);
    f64 auto_frac = 0.4;
    if (method == EdgeMethod::kRoberts) {
        static constexpr f64 kRx[4] = {1.0, 0.0, 0.0, -1.0};
        static constexpr f64 kRy[4] = {0.0, -1.0, 1.0, 0.0};
        (void)imfilter_2d(img_col_major, height, width, ArrayRef<f64>(kRx, 4), 2, 2,
                          PadMode::kReplicate, 0.0, false, false, gx.as_mut_ref());
        (void)imfilter_2d(img_col_major, height, width, ArrayRef<f64>(kRy, 4), 2, 2,
                          PadMode::kReplicate, 0.0, false, false, gy.as_mut_ref());
        auto_frac = 0.5;
    } else {
        const GradientMethod gm = (method == EdgeMethod::kPrewitt) ? GradientMethod::kPrewitt
                                                                   : GradientMethod::kSobel;
        (void)imgradientxy_2d(img_col_major, height, width, gm, gx.as_mut_ref(), gy.as_mut_ref());
    }

    Array<f64> gmag(a);
    gmag.resize(total);
    for (usize i = 0; i < total; ++i) {
        gmag[i] = sqrt(gx[i] * gx[i] + gy[i] * gy[i]);
    }
    const f64 th = is_nan(thresh) ? (auto_frac * max_abs(gmag.as_ref())) : thresh;
    for (usize i = 0; i < total; ++i) {
        out_col_major[i] = (gmag[i] > th) ? u8{1} : u8{0};
    }
    return ImageStatus::kOk;
}

ImageStatus cornermetric_2d(ArrayRef<f64> unit_col_major, usize height, usize width,
                            CornerMethod method, f64 sensitivity_factor, ArrayRef<f64> filter_coef,
                            MutArrayRef<f64> out_col_major, Allocator* alloc) {
    const usize total = height * width;
    if (height < 3 || width < 3 || unit_col_major.size() != total
        || out_col_major.size() != total) {
        return ImageStatus::kInvalidArgument;
    }
    if (method == CornerMethod::kHarris
        && (!(sensitivity_factor > 0.0) || !(sensitivity_factor < 0.25))) {
        return ImageStatus::kInvalidArgument;
    }

    static constexpr f64 kDefaultCoef[5] = {0.12007838422146333, 0.23388075753362165,
                                            0.29208171648983004, 0.23388075753362165,
                                            0.12007838422146333};
    ArrayRef<f64> fc =
        (filter_coef.size() == 0) ? ArrayRef<f64>(kDefaultCoef, 5) : filter_coef;
    const usize lc = fc.size();
    if (lc < 3 || (lc % 2) == 0) {
        return ImageStatus::kInvalidArgument;
    }
    const usize half_lc = lc / 2;
    const usize removed = half_lc - 1;

    Allocator* a = alloc ? alloc : default_allocator();
    Array<f64> dx(a);
    dx.resize(total);
    Array<f64> dy(a);
    dy.resize(total);
    static constexpr f64 kDiff[3] = {-1.0, 0.0, 1.0};
    (void)imfilter_2d(unit_col_major, height, width, ArrayRef<f64>(kDiff, 3), 1, 3,
                      PadMode::kReplicate, 0.0, false, true, dx.as_mut_ref());
    (void)imfilter_2d(unit_col_major, height, width, ArrayRef<f64>(kDiff, 3), 3, 1,
                      PadMode::kReplicate, 0.0, false, true, dy.as_mut_ref());

    const usize ht = height - 2;
    const usize wt = width - 2;
    const usize nt = ht * wt;
    Array<f64> amat(a);
    amat.resize(nt);
    Array<f64> bmat(a);
    bmat.resize(nt);
    Array<f64> cmat(a);
    cmat.resize(nt);
    for (usize c = 0; c < wt; ++c) {
        for (usize r = 0; r < ht; ++r) {
            const usize src = (c + 1) * height + (r + 1);
            const f64 dxv = dx[src];
            const f64 dyv = dy[src];
            const usize dst = c * ht + r;
            amat[dst] = dxv * dxv;
            bmat[dst] = dyv * dyv;
            cmat[dst] = dxv * dyv;
        }
    }

    Array<f64> wk(a);
    wk.resize(lc * lc);
    for (usize c = 0; c < lc; ++c) {
        for (usize r = 0; r < lc; ++r) {
            wk[c * lc + r] = fc[r] * fc[c];
        }
    }

    const usize hp = ht + 2 * half_lc;
    const usize wp = wt + 2 * half_lc;
    Array<f64> padded(a);
    padded.resize(hp * wp);
    Array<f64> filt(a);
    filt.resize(hp * wp);

    auto smooth_and_crop = [&](ArrayRef<f64> src, MutArrayRef<f64> dst_hw) {
        const i32 ph = as<i32>(half_lc);
        (void)padarray_f64(src, ht, wt, 1, ph, ph, ph, ph, 0, 0, PadMode::kReplicate, 0.0,
                           padded.as_mut_ref());
        (void)imfilter_2d(padded.as_ref(), hp, wp, wk.as_ref(), lc, lc, PadMode::kReplicate, 0.0,
                          false, true, filt.as_mut_ref());
        for (usize c = 0; c < width; ++c) {
            for (usize r = 0; r < height; ++r) {
                dst_hw[c * height + r] = filt[(c + removed) * hp + (r + removed)];
            }
        }
    };

    Array<f64> af(a);
    af.resize(total);
    Array<f64> bf(a);
    bf.resize(total);
    Array<f64> cf(a);
    cf.resize(total);
    smooth_and_crop(amat.as_ref(), af.as_mut_ref());
    smooth_and_crop(bmat.as_ref(), bf.as_mut_ref());
    smooth_and_crop(cmat.as_ref(), cf.as_mut_ref());

    if (method == CornerMethod::kHarris) {
        for (usize i = 0; i < total; ++i) {
            const f64 s = af[i] + bf[i];
            out_col_major[i] = af[i] * bf[i] - cf[i] * cf[i] - sensitivity_factor * s * s;
        }
    } else {
        for (usize i = 0; i < total; ++i) {
            const f64 s = af[i] + bf[i];
            const f64 d = af[i] - bf[i];
            out_col_major[i] = (s - sqrt(d * d + 4.0 * cf[i] * cf[i])) * 0.5;
        }
    }
    return ImageStatus::kOk;
}

ImageStatus corner_peaks_2d(ArrayRef<f64> metric_col_major, usize height, usize width, i32 max_n,
                            f64 quality_level, Array<BoundaryPoint>& out_peaks, Allocator* alloc) {
    const usize total = height * width;
    if (metric_col_major.size() != total || !is_finite(quality_level) ||
        !(quality_level >= 0.0 && quality_level <= 1.0)) {
        return ImageStatus::kInvalidArgument;
    }
    out_peaks.clear();
    if (total == 0) {
        return ImageStatus::kOk;
    }
    f64 mx = -Limits<f64>::infinity();
    for (usize i = 0; i < total; ++i) {
        const f64 v = metric_col_major[i];
        if (is_finite(v) && v > mx) {
            mx = v;
        }
    }
    if (!(mx > 0.0) || !is_finite(mx)) {
        return ImageStatus::kOk;
    }
    const f64 thr = quality_level * mx;
    Allocator* a = alloc ? alloc : default_allocator();
    auto at = [&](usize r, usize c) noexcept -> f64 { return metric_col_major[c * height + r]; };

    Array<u8> is_max(a);
    is_max.assign(total, u8{0});
    for (usize c = 0; c < width; ++c) {
        for (usize r = 0; r < height; ++r) {
            const f64 v = at(r, c);
            if (!(v > thr)) {
                continue;
            }
            bool peak = true;
            for (i32 dc = -1; dc <= 1 && peak; ++dc) {
                for (i32 dr = -1; dr <= 1 && peak; ++dr) {
                    if (dr == 0 && dc == 0) {
                        continue;
                    }
                    const i64 rr = as<i64>(r) + dr;
                    const i64 cc = as<i64>(c) + dc;
                    if (rr < 0 || cc < 0 || rr >= as<i64>(height) || cc >= as<i64>(width)) {
                        continue;
                    }
                    if (at(as<usize>(rr), as<usize>(cc)) > v) {
                        peak = false;
                    }
                }
            }
            if (peak) {
                is_max[c * height + r] = 1;
            }
        }
    }

    const f64 quantum = mx * 1e-9;
    struct Peak {
        f64 row;
        f64 col;
        i64 key;
    };
    Array<Peak> peaks(a);
    Array<u8> seen(a);
    seen.assign(total, u8{0});
    Array<usize> stack(a);
    for (usize idx = 0; idx < total; ++idx) {
        if (!is_max[idx] || seen[idx]) {
            continue;
        }
        f64 sum_r = 0.0;
        f64 sum_c = 0.0;
        f64 best = metric_col_major[idx];
        usize cnt = 0;
        stack.clear();
        stack.push_back(idx);
        seen[idx] = 1;
        while (!stack.empty()) {
            const usize p = stack.back();
            stack.pop_back();
            const usize pr = p % height;
            const usize pc = p / height;
            sum_r += as<f64>(pr);
            sum_c += as<f64>(pc);
            ++cnt;
            if (metric_col_major[p] > best) {
                best = metric_col_major[p];
            }
            for (i32 dc = -1; dc <= 1; ++dc) {
                for (i32 dr = -1; dr <= 1; ++dr) {
                    if (dr == 0 && dc == 0) {
                        continue;
                    }
                    const i64 rr = as<i64>(pr) + dr;
                    const i64 cc = as<i64>(pc) + dc;
                    if (rr < 0 || cc < 0 || rr >= as<i64>(height) || cc >= as<i64>(width)) {
                        continue;
                    }
                    const usize q = as<usize>(cc) * height + as<usize>(rr);
                    if (is_max[q] && !seen[q]) {
                        seen[q] = 1;
                        stack.push_back(q);
                    }
                }
            }
        }
        peaks.push_back(Peak{sum_r / as<f64>(cnt), sum_c / as<f64>(cnt),
                             round_i64(best / quantum)});
    }

    stable_sort(peaks.begin(), peaks.end(),
                [](const Peak& x, const Peak& y) noexcept { return x.key > y.key; });
    usize k_out = peaks.size();
    if (max_n >= 0 && as<usize>(max_n) < k_out) {
        k_out = as<usize>(max_n);
    }
    out_peaks.reserve(k_out);
    for (usize i = 0; i < k_out; ++i) {
        out_peaks.push_back(BoundaryPoint{round_i64(peaks[i].row), round_i64(peaks[i].col)});
    }
    return ImageStatus::kOk;
}

ImageStatus hough_2d(ArrayRef<u8> bw_col_major, usize height, usize width, f64 rho_res,
                     ArrayRef<f64> theta_deg, Array<f64>& out_h_col_major, Array<f64>& out_rho,
                     Allocator* alloc) {
    if (bw_col_major.size() != height * width || !(rho_res > 0.0) || !is_finite(rho_res)
        || theta_deg.size() == 0) {
        return ImageStatus::kInvalidArgument;
    }
    const usize ntheta = theta_deg.size();
    for (usize k = 0; k < ntheta; ++k) {
        if (!is_finite(theta_deg[k])) {
            return ImageStatus::kInvalidArgument;
        }
    }
    const f64 d_diag = (height > 0 && width > 0)
                           ? sqrt(as<f64>(height - 1) * as<f64>(height - 1)
                                  + as<f64>(width - 1) * as<f64>(width - 1))
                           : 0.0;
    const usize q = (d_diag == 0.0) ? 0 : as<usize>(ceil(d_diag / rho_res));
    const usize nrho = 2 * q + 1;

    out_rho.resize(nrho);
    if (nrho == 1) {
        out_rho[0] = 0.0;
    } else {
        const f64 lo = -as<f64>(q) * rho_res;
        for (usize i = 0; i < nrho; ++i) {
            out_rho[i] = lo + rho_res * as<f64>(i);
        }
    }

    Allocator* a = alloc ? alloc : default_allocator();
    constexpr f64 kDeg2Rad = kPi / 180.0;
    Array<f64> cos_t(a);
    cos_t.resize(ntheta);
    Array<f64> sin_t(a);
    sin_t.resize(ntheta);
    for (usize k = 0; k < ntheta; ++k) {
        cos_t[k] = cos(theta_deg[k] * kDeg2Rad);
        sin_t[k] = sin(theta_deg[k] * kDeg2Rad);
    }

    out_h_col_major.assign(nrho * ntheta, 0.0);
    for (usize c = 0; c < width; ++c) {
        for (usize r = 0; r < height; ++r) {
            if (!bw_col_major[c * height + r]) {
                continue;
            }
            const f64 x = as<f64>(c);
            const f64 y = as<f64>(r);
            for (usize tk = 0; tk < ntheta; ++tk) {
                const f64 rho_val = x * cos_t[tk] + y * sin_t[tk];
                const i64 bin = round_i64(rho_val / rho_res) + as<i64>(q);
                if (bin < 0 || as<usize>(bin) >= nrho) {
                    continue;
                }
                out_h_col_major[tk * nrho + as<usize>(bin)] += 1.0;
            }
        }
    }
    return ImageStatus::kOk;
}

ImageStatus houghpeaks_2d(ArrayRef<f64> h_col_major, usize nrho, usize ntheta, usize num_peaks,
                          f64 threshold, usize nhood_rho, usize nhood_theta,
                          ArrayRef<f64> theta_deg, Array<BoundaryPoint>& out_peaks,
                          Allocator* alloc) {
    const usize total = nrho * ntheta;
    if (h_col_major.size() != total) {
        return ImageStatus::kInvalidArgument;
    }
    out_peaks.clear();
    if (total == 0 || num_peaks == 0) {
        return ImageStatus::kOk;
    }
    auto next_odd = [](usize v) noexcept -> usize {
        if (v < 1) v = 1;
        if ((v % 2) == 0) v += 1;
        return v;
    };
    if (nhood_rho == 0) nhood_rho = next_odd((nrho + 49) / 50);
    if (nhood_theta == 0) nhood_theta = next_odd((ntheta + 49) / 50);
    if ((nhood_rho % 2) == 0) ++nhood_rho;
    if ((nhood_theta % 2) == 0) ++nhood_theta;

    f64 mx_h = -Limits<f64>::infinity();
    for (usize i = 0; i < total; ++i) {
        if (is_finite(h_col_major[i]) && h_col_major[i] > mx_h) {
            mx_h = h_col_major[i];
        }
    }
    if (!is_finite(mx_h)) {
        return ImageStatus::kOk;
    }
    if (is_nan(threshold) || threshold < 0.0) {
        threshold = 0.5 * mx_h;
    }

    bool is_antisym = false;
    if (theta_deg.size() >= 2) {
        f64 min_t = theta_deg[0];
        f64 max_t = theta_deg[0];
        for (usize i = 1; i < theta_deg.size(); ++i) {
            if (theta_deg[i] < min_t) min_t = theta_deg[i];
            if (theta_deg[i] > max_t) max_t = theta_deg[i];
        }
        const f64 t_res = abs(max_t - min_t) / as<f64>(theta_deg.size() - 1);
        is_antisym = abs(min_t + t_res * as<f64>(nhood_theta)) <= max_t;
    } else if (theta_deg.size() == 0) {
        is_antisym = abs(-90.0 + as<f64>(nhood_theta)) <= 89.0;
    }

    Allocator* a = alloc ? alloc : default_allocator();
    Array<f64> hwork(a);
    hwork.resize(total);
    copy_n(hwork.data(), h_col_major.data(), total);

    const i64 half_r = as<i64>(nhood_rho / 2);
    const i64 half_t = as<i64>(nhood_theta / 2);
    out_peaks.reserve(num_peaks);

    while (out_peaks.size() < num_peaks) {
        usize max_idx = 0;
        f64 max_v = -Limits<f64>::infinity();
        for (usize i = 0; i < total; ++i) {
            if (is_finite(hwork[i]) && hwork[i] > max_v) {
                max_v = hwork[i];
                max_idx = i;
            }
        }
        if (!(max_v >= threshold)) {
            break;
        }
        const usize p = max_idx % nrho;
        const usize q = max_idx / nrho;
        out_peaks.push_back(BoundaryPoint{as<i64>(p), as<i64>(q)});

        const i64 p1 = as<i64>(p) - half_r;
        const i64 p2 = as<i64>(p) + half_r;
        const i64 q1 = as<i64>(q) - half_t;
        const i64 q2 = as<i64>(q) + half_t;
        for (i64 qq = q1; qq <= q2; ++qq) {
            for (i64 pp = p1; pp <= p2; ++pp) {
                if (pp < 0 || pp >= as<i64>(nrho)) {
                    continue;
                }
                i64 q_wrap = qq;
                i64 p_wrap = pp;
                if (q_wrap < 0 || q_wrap >= as<i64>(ntheta)) {
                    if (is_antisym) {
                        if (q_wrap < 0) {
                            q_wrap += as<i64>(ntheta);
                        } else {
                            q_wrap -= as<i64>(ntheta);
                        }
                        p_wrap = as<i64>(nrho) - 1 - pp;
                        if (p_wrap < 0 || p_wrap >= as<i64>(nrho) || q_wrap < 0
                            || q_wrap >= as<i64>(ntheta)) {
                            continue;
                        }
                    } else {
                        continue;
                    }
                }
                hwork[as<usize>(q_wrap) * nrho + as<usize>(p_wrap)] = 0.0;
            }
        }
    }
    return ImageStatus::kOk;
}

ImageStatus houghlines_2d(ArrayRef<u8> bw_col_major, usize height, usize width,
                          ArrayRef<f64> theta_deg, ArrayRef<f64> rho,
                          ArrayRef<BoundaryPoint> peaks_0based, f64 fill_gap, f64 min_length,
                          Array<HoughLine>& out_lines, Allocator* alloc) {
    if (bw_col_major.size() != height * width || rho.size() < 2 || theta_deg.size() == 0
        || !(fill_gap > 0.0) || !is_finite(fill_gap) || !(min_length > 0.0)
        || !is_finite(min_length)) {
        return ImageStatus::kInvalidArgument;
    }
    out_lines.clear();
    Allocator* a = alloc ? alloc : default_allocator();

    Array<f64> px(a);
    Array<f64> py(a);
    px.reserve(height * width / 16 + 4);
    py.reserve(height * width / 16 + 4);
    for (usize c = 0; c < width; ++c) {
        for (usize r = 0; r < height; ++r) {
            if (bw_col_major[c * height + r]) {
                px.push_back(as<f64>(c));
                py.push_back(as<f64>(r));
            }
        }
    }
    const usize np = px.size();
    if (np == 0) {
        return ImageStatus::kOk;
    }

    constexpr f64 kDeg2Rad = kPi / 180.0;
    const f64 rho_first = rho[0];
    const f64 rho_last = rho[rho.size() - 1];
    if (!is_finite(rho_first) || !is_finite(rho_last) || rho_last == rho_first) {
        return ImageStatus::kInvalidArgument;
    }
    const f64 slope = as<f64>(rho.size() - 1) / (rho_last - rho_first);
    const f64 fillgap_sq = fill_gap * fill_gap;
    const f64 minlen_sq = min_length * min_length;

    Array<usize> idx(a);
    Array<usize> rseg(a);
    Array<usize> cseg(a);
    Array<usize> order(a);
    Array<i64> xs(a);
    Array<i64> ys(a);
    Array<usize> splits(a);

    for (usize k = 0; k < peaks_0based.size(); ++k) {
        const i64 rbin = peaks_0based[k].row;
        const i64 cbin = peaks_0based[k].col;
        if (cbin < 0 || cbin >= as<i64>(theta_deg.size()) || rbin < 0
            || rbin >= as<i64>(rho.size())) {
            continue;
        }
        const f64 theta_c = theta_deg[as<usize>(cbin)] * kDeg2Rad;
        const f64 cos_t = cos(theta_c);
        const f64 sin_t = sin(theta_c);

        idx.clear();
        for (usize i = 0; i < np; ++i) {
            const f64 rho_xy = px[i] * cos_t + py[i] * sin_t;
            const i64 bin = round_i64(slope * (rho_xy - rho_first));
            if (bin == rbin) {
                idx.push_back(i);
            }
        }
        if (idx.empty()) {
            continue;
        }

        rseg.clear();
        cseg.clear();
        for (usize i = 0; i < idx.size(); ++i) {
            rseg.push_back(as<usize>(py[idx[i]]));
            cseg.push_back(as<usize>(px[idx[i]]));
        }
        usize rmin = rseg[0];
        usize rmax = rseg[0];
        usize cmin = cseg[0];
        usize cmax = cseg[0];
        for (usize i = 1; i < rseg.size(); ++i) {
            if (rseg[i] < rmin) rmin = rseg[i];
            if (rseg[i] > rmax) rmax = rseg[i];
            if (cseg[i] < cmin) cmin = cseg[i];
            if (cseg[i] > cmax) cmax = cseg[i];
        }
        const usize r_range = rmax - rmin;
        const usize c_range = cmax - cmin;
        order.clear();
        for (usize i = 0; i < rseg.size(); ++i) {
            order.push_back(i);
        }
        if (r_range > c_range) {
            sort(order.begin(), order.end(), [&](usize x, usize y) noexcept {
                if (rseg[x] != rseg[y]) return rseg[x] < rseg[y];
                return cseg[x] < cseg[y];
            });
        } else {
            sort(order.begin(), order.end(), [&](usize x, usize y) noexcept {
                if (cseg[x] != cseg[y]) return cseg[x] < cseg[y];
                return rseg[x] < rseg[y];
            });
        }

        xs.clear();
        ys.clear();
        for (usize i = 0; i < order.size(); ++i) {
            xs.push_back(as<i64>(cseg[order[i]]));
            ys.push_back(as<i64>(rseg[order[i]]));
        }

        splits.clear();
        splits.push_back(0);
        for (usize i = 0; i + 1 < xs.size(); ++i) {
            const f64 dx = as<f64>(xs[i + 1] - xs[i]);
            const f64 dy = as<f64>(ys[i + 1] - ys[i]);
            if (dx * dx + dy * dy > fillgap_sq) {
                splits.push_back(i + 1);
            }
        }
        splits.push_back(xs.size());

        for (usize p = 0; p + 1 < splits.size(); ++p) {
            const usize lo = splits[p];
            const usize hi = splits[p + 1];
            if (hi <= lo) {
                continue;
            }
            const f64 p1x = as<f64>(xs[lo]);
            const f64 p1y = as<f64>(ys[lo]);
            const f64 p2x = as<f64>(xs[hi - 1]);
            const f64 p2y = as<f64>(ys[hi - 1]);
            const f64 d2 = (p2x - p1x) * (p2x - p1x) + (p2y - p1y) * (p2y - p1y);
            if (d2 >= minlen_sq) {
                out_lines.push_back(HoughLine{p1x, p1y, p2x, p2y, theta_deg[as<usize>(cbin)],
                                              rho[as<usize>(rbin)]});
            }
        }
    }
    return ImageStatus::kOk;
}

ImageStatus imfindcircles_2d(ArrayRef<f64> gray_col_major, usize height, usize width, f64 rmin,
                             f64 rmax, CirclePolarity polarity, CircleMethod method,
                             f64 sensitivity, f64 edge_threshold, bool use_single_grad,
                             Array<CircleDetection>& out_circles, Allocator* alloc) {
    const usize total = height * width;
    if (height <= 1 || width <= 1 || gray_col_major.size() != total || !is_finite(rmin)
        || !is_finite(rmax) || !(rmin > 0.0) || !(rmax >= rmin) || !is_finite(sensitivity)
        || sensitivity < 0.0 || sensitivity > 1.0 || is_nan(edge_threshold)
        || edge_threshold > 1.0) {
        return ImageStatus::kInvalidArgument;
    }
    out_circles.clear();

    // Flat image check.
    bool is_flat = true;
    for (usize i = 1; i < total; ++i) {
        if (gray_col_major[i] != gray_col_major[0]) {
            is_flat = false;
            break;
        }
    }
    if (is_flat) {
        return ImageStatus::kOk;
    }

    Allocator* a = alloc ? alloc : default_allocator();
    // Sobel gradient (convolved with replicate padding, matching chaccum).
    static constexpr f64 kHx[9] = {-1, -2, -1, 0, 0, 0, 1, 2, 1};
    static constexpr f64 kHy[9] = {-1, 0, 1, -2, 0, 2, -1, 0, 1};
    Array<f64> gx(a);
    gx.resize(total);
    Array<f64> gy(a);
    gy.resize(total);
    (void)imfilter_2d(gray_col_major, height, width, ArrayRef<f64>(kHx, 9), 3, 3,
                      PadMode::kReplicate, 0.0, false, true, gx.as_mut_ref());
    (void)imfilter_2d(gray_col_major, height, width, ArrayRef<f64>(kHy, 9), 3, 3,
                      PadMode::kReplicate, 0.0, false, true, gy.as_mut_ref());

    Array<f64> grad_mag(a);
    grad_mag.resize(total);
    f64 gmax = 0.0;
    for (usize i = 0; i < total; ++i) {
        const f64 gm = use_single_grad ? as<f64>(hypot(as<f32>(gx[i]), as<f32>(gy[i])))
                                       : hypot(gx[i], gy[i]);
        grad_mag[i] = gm;
        if (gm > gmax) {
            gmax = gm;
        }
    }
    if (!(gmax > 0.0)) {
        return ImageStatus::kOk;
    }

    f64 eth = edge_threshold;
    if (eth < 0.0) {
        Array<f64> gnorm(a);
        gnorm.resize(total);
        if (use_single_grad) {
            const f32 gmax_f = as<f32>(gmax);
            for (usize i = 0; i < total; ++i) {
                gnorm[i] = as<f64>(as<f32>(grad_mag[i]) / gmax_f);
            }
        } else {
            for (usize i = 0; i < total; ++i) {
                gnorm[i] = grad_mag[i] / gmax;
            }
        }
        StaticArray<f64, 256> counts{};
        (void)histogram(gnorm.as_ref(), 0.0, 1.0, MutArrayRef<f64>(counts.data(), 256));
        eth = otsu_threshold(ArrayRef<f64>(counts.data(), 256), a).threshold;
    }

    const f64 t = use_single_grad ? as<f64>(as<f32>(gmax) * as<f32>(eth)) : (gmax * eth);

    struct EdgePix {
        f64 ex;
        f64 ey;
        f64 ux;
        f64 uy;
    };
    Array<EdgePix> edges(a);
    edges.reserve(total / 8 + 4);
    for (usize c = 0; c < width; ++c) {
        for (usize r = 0; r < height; ++r) {
            const usize idx = c * height + r;
            if (is_finite(grad_mag[idx]) && grad_mag[idx] > t) {
                const f64 ux = use_single_grad
                                   ? as<f64>(as<f32>(gx[idx]) / as<f32>(grad_mag[idx]))
                                   : (gx[idx] / grad_mag[idx]);
                const f64 uy = use_single_grad
                                   ? as<f64>(as<f32>(gy[idx]) / as<f32>(grad_mag[idx]))
                                   : (gy[idx] / grad_mag[idx]);
                if (is_finite(ux) && is_finite(uy)) {
                    edges.push_back(EdgePix{as<f64>(c + 1), as<f64>(r + 1), ux, uy});
                }
            }
        }
    }
    if (edges.empty()) {
        return ImageStatus::kOk;
    }

    Array<f64> rad_vec(a);
    if (rmin < rmax) {
        for (f64 r = rmin; r <= rmax + 1e-9; r += 0.5) {
            rad_vec.push_back(r);
        }
    } else {
        rad_vec.push_back(rmin);
    }
    const usize nr = rad_vec.size();
    const f64 pol_sign = (polarity == CirclePolarity::kBright) ? 1.0 : -1.0;

    Array<f64> rr_signed(a);
    rr_signed.resize(nr);
    Array<f64> w0_re(a);
    w0_re.resize(nr);
    Array<f64> w0_im(a);
    w0_im.assign(nr, 0.0);
    const f64 ln_r1 = log(rad_vec.front());
    const f64 ln_rend = log(rad_vec.back());
    for (usize k = 0; k < nr; ++k) {
        rr_signed[k] = pol_sign * rad_vec[k];
        const f64 inv_circ = 1.0 / (2.0 * kPi * rad_vec[k]);
        if (method == CircleMethod::kPhaseCode && nr > 1) {
            const f64 phi = ((log(rad_vec[k]) - ln_r1) / (ln_rend - ln_r1) * 2.0 * kPi) - kPi;
            w0_re[k] = cos(phi) * inv_circ;
            w0_im[k] = sin(phi) * inv_circ;
        } else {
            w0_re[k] = inv_circ;
            w0_im[k] = 0.0;
        }
    }

    Array<f64> accum_re(a);
    accum_re.assign(total, 0.0);
    Array<f64> accum_im(a);
    accum_im.assign(total, 0.0);
    const i64 ml = as<i64>(height);
    const i64 nl = as<i64>(width);
    for (usize ei = 0; ei < edges.size(); ++ei) {
        const EdgePix& ep = edges[ei];
        for (usize k = 0; k < nr; ++k) {
            const i64 xc = round_i64(ep.ex + (-rr_signed[k]) * ep.ux);
            const i64 yc = round_i64(ep.ey + (-rr_signed[k]) * ep.uy);
            if (xc >= 1 && xc <= nl && yc >= 1 && yc <= ml) {
                const usize aidx = as<usize>(xc - 1) * height + as<usize>(yc - 1);
                accum_re[aidx] += w0_re[k];
                accum_im[aidx] += w0_im[k];
            }
        }
    }

    Array<f64> accum_abs(a);
    accum_abs.resize(total);
    bool any_nz = false;
    for (usize i = 0; i < total; ++i) {
        accum_abs[i] = hypot(accum_re[i], accum_im[i]);
        if (accum_abs[i] != 0.0) {
            any_nz = true;
        }
    }
    if (!any_nz) {
        return ImageStatus::kOk;
    }
    bool accum_flat = true;
    for (usize i = 1; i < total; ++i) {
        if (accum_abs[i] != accum_abs[0]) {
            accum_flat = false;
            break;
        }
    }
    if (accum_flat) {
        return ImageStatus::kOk;
    }

    Array<f64> hd_in(a);
    hd_in.resize(total);
    if (min(height, width) > 5) {
        StaticArray<f64, 25> win{};
        for (usize c = 0; c < width; ++c) {
            for (usize r = 0; r < height; ++r) {
                usize k = 0;
                for (i32 dc = -2; dc <= 2; ++dc) {
                    const i64 nc = as<i64>(c) + dc;
                    for (i32 dr = -2; dr <= 2; ++dr) {
                        const i64 nr_ = as<i64>(r) + dr;
                        if (nr_ < 0 || nr_ >= ml || nc < 0 || nc >= nl) {
                            win[k++] = 0.0;
                        } else {
                            win[k++] = accum_abs[as<usize>(nc) * height + as<usize>(nr_)];
                        }
                    }
                }
                nth_element(win.begin(), win.begin() + 12, win.end());
                hd_in[c * height + r] = win[12];
            }
        }
    } else {
        copy_n(hd_in.data(), accum_abs.data(), total);
    }

    const f64 accum_thresh = 1.0 - sensitivity;
    const f64 eps_t = (accum_thresh > 0.0)
                          ? (nextafter(accum_thresh, Limits<f64>::infinity()) - accum_thresh)
                          : 0.0;
    const f64 supp_thresh = max(accum_thresh - eps_t, 0.0);
    Array<f64> hd(a);
    hd.resize(total);
    (void)imhmax_2d(hd_in.as_ref(), height, width, supp_thresh, 8, hd.as_mut_ref(), a);

    static constexpr i32 kDr8[8] = {-1, 0, 1, -1, 1, -1, 0, 1};
    static constexpr i32 kDc8[8] = {-1, -1, -1, 0, 0, 1, 1, 1};
    Array<u8> visited(a);
    visited.assign(total, u8{0});
    Array<usize> comp(a);
    Array<usize> q(a);
    comp.reserve(32);
    q.reserve(32);

    struct Cand {
        f64 xc;
        f64 yc;
        f64 metric;
    };
    Array<Cand> cands(a);
    for (usize i = 0; i < total; ++i) {
        if (visited[i]) {
            continue;
        }
        visited[i] = 1;
        comp.clear();
        q.clear();
        comp.push_back(i);
        q.push_back(i);
        const f64 v0 = hd[i];
        bool has_higher = false;
        usize head = 0;
        while (head < q.size()) {
            const usize curr = q[head++];
            const i64 r = as<i64>(curr % height);
            const i64 c = as<i64>(curr / height);
            for (i32 k = 0; k < 8; ++k) {
                const i64 nr_ = r + kDr8[k];
                const i64 nc = c + kDc8[k];
                if (nr_ < 0 || nr_ >= ml || nc < 0 || nc >= nl) {
                    continue;
                }
                const usize nb = as<usize>(nc) * height + as<usize>(nr_);
                if (hd[nb] > v0) {
                    has_higher = true;
                } else if (hd[nb] == v0 && !visited[nb]) {
                    visited[nb] = 1;
                    comp.push_back(nb);
                    q.push_back(nb);
                }
            }
        }
        if (!has_higher) {
            f64 wsum = 0.0;
            f64 xsum = 0.0;
            f64 ysum = 0.0;
            for (usize j = 0; j < comp.size(); ++j) {
                const usize idx = comp[j];
                const f64 w = accum_abs[idx];
                const f64 r1 = as<f64>(idx % height) + 1.0;
                const f64 c1 = as<f64>(idx / height) + 1.0;
                wsum += w;
                xsum += c1 * w;
                ysum += r1 * w;
            }
            if (wsum > 0.0) {
                const f64 xc = xsum / wsum;
                const f64 yc = ysum / wsum;
                const i64 r_idx = clamp<i64>(round_i64(yc) - 1, 0, ml - 1);
                const i64 c_idx = clamp<i64>(round_i64(xc) - 1, 0, nl - 1);
                const f64 met = hd[as<usize>(c_idx) * height + as<usize>(r_idx)];
                if (met >= accum_thresh) {
                    cands.push_back(Cand{xc, yc, met});
                }
            }
        }
    }
    if (cands.empty()) {
        return ImageStatus::kOk;
    }

    stable_sort(cands.begin(), cands.end(),
                [](const Cand& x, const Cand& y) noexcept { return x.metric > y.metric; });

    out_circles.reserve(cands.size());
    for (usize k = 0; k < cands.size(); ++k) {
        f64 rad = rmin;
        if (rmin == rmax) {
            rad = rmin;
        } else if (method == CircleMethod::kPhaseCode) {
            const i64 r_idx = clamp<i64>(round_i64(cands[k].yc) - 1, 0, ml - 1);
            const i64 c_idx = clamp<i64>(round_i64(cands[k].xc) - 1, 0, nl - 1);
            const usize aidx = as<usize>(c_idx) * height + as<usize>(r_idx);
            const f64 cen_phase = atan2(accum_im[aidx], accum_re[aidx]);
            rad = exp(((cen_phase + kPi) / (2.0 * kPi) * (ln_rend - ln_r1)) + ln_r1);
        } else {
            const i64 left = max<i64>(as<i64>(floor(cands[k].xc - rmax)), 1);
            const i64 right = min<i64>(as<i64>(ceil(cands[k].xc + rmax)), nl);
            const i64 top = max<i64>(as<i64>(floor(cands[k].yc - rmax)), 1);
            const i64 bottom = min<i64>(as<i64>(ceil(cands[k].yc + rmax)), ml);
            const i64 r1_int = round_i64(rmin);
            const i64 r2_int = round_i64(rmax);
            const usize nbins = as<usize>(r2_int - r1_int + 1);
            Array<f64> hbin(a);
            hbin.assign(nbins, 0.0);
            for (i64 cc = left; cc <= right; ++cc) {
                for (i64 rr = top; rr <= bottom; ++rr) {
                    const i64 rbin = round_i64(
                        hypot(as<f64>(cc) - cands[k].xc, as<f64>(rr) - cands[k].yc));
                    if (rbin >= r1_int && rbin <= r2_int) {
                        hbin[as<usize>(rbin - r1_int)] +=
                            grad_mag[as<usize>(cc - 1) * height + as<usize>(rr - 1)];
                    }
                }
            }
            f64 best_h = -1.0;
            i64 best_r = r1_int;
            for (usize b = 0; b < nbins; ++b) {
                const f64 rcur = as<f64>(r1_int + as<i64>(b));
                if (!(rcur > 0.0)) {
                    continue;
                }
                const f64 hnorm = hbin[b] / (2.0 * kPi * rcur);
                if (hnorm > best_h) {
                    best_h = hnorm;
                    best_r = r1_int + as<i64>(b);
                }
            }
            rad = as<f64>(best_r);
        }
        // Output 0-based (col, row) coordinates (caller adds +1.0 for 1-based convention).
        out_circles.push_back(CircleDetection{cands[k].xc - 1.0, cands[k].yc - 1.0, rad,
                                              cands[k].metric});
    }
    return ImageStatus::kOk;
}

} // namespace keel::image
