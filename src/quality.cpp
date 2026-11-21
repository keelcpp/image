#include "keel/image.h"

#include "keel/arena.h"
#include "keel/array.h"
#include "keel/math.h"

namespace keel::image {

f64 mse(ArrayRef<f64> a, ArrayRef<f64> b) noexcept {
    const usize n = keel::min(a.size(), b.size());
    if (n == 0) return 0.0;
    f64 sum = 0.0;
    for (usize i = 0; i < n; ++i) {
        const f64 d = a[i] - b[i];
        sum += d * d;
    }
    return sum / static_cast<f64>(n);
}

f64 mse_c64(ArrayRef<c64> a, ArrayRef<c64> b) noexcept {
    const usize n = keel::min(a.size(), b.size());
    if (n == 0) return 0.0;
    f64 sum = 0.0;
    for (usize i = 0; i < n; ++i) {
        const c64 d = a[i] - b[i];
        sum += d.real() * d.real() + d.imag() * d.imag();
    }
    return sum / static_cast<f64>(n);
}

f64 psnr(ArrayRef<f64> a, ArrayRef<f64> b, f64 peak) noexcept {
    const usize n = keel::min(a.size(), b.size());
    if (n == 0) return Limits<f64>::infinity();
    const f64 m = mse(a, b);
    if (m == 0.0) return Limits<f64>::infinity();
    return 10.0 * keel::log10((peak * peak) / m);
}

f64 mean2(ArrayRef<f64> img) noexcept {
    const usize n = img.size();
    if (n == 0) return Limits<f64>::quiet_nan();
    f64 s = 0.0;
    for (usize i = 0; i < n; ++i) {
        s += img[i];
    }
    return s / static_cast<f64>(n);
}

f64 std2(ArrayRef<f64> img) noexcept {
    const usize n = img.size();
    if (n == 0) return Limits<f64>::quiet_nan();
    if (n == 1) return 0.0;
    const f64 mu = mean2(img);
    f64 v = 0.0;
    for (usize i = 0; i < n; ++i) {
        const f64 d = img[i] - mu;
        v += d * d;
    }
    return keel::sqrt(v / static_cast<f64>(n - 1));
}

ImageStatus gaussian_kernel_2d(usize height, usize width, f64 sigma,
                               MutArrayRef<f64> out_col_major) noexcept {
    if (height == 0 || width == 0 || !(sigma > 0.0) || !keel::is_finite(sigma)) {
        return ImageStatus::kInvalidArgument;
    }
    const usize total = height * width;
    if (out_col_major.size() < total) {
        return ImageStatus::kInvalidArgument;
    }

    const f64 cy = 0.5 * static_cast<f64>(height - 1);
    const f64 cx = 0.5 * static_cast<f64>(width - 1);
    const f64 inv_2s2 = 1.0 / (2.0 * sigma * sigma);
    f64 sum = 0.0;
    for (usize c = 0; c < width; ++c) {
        const f64 dx = static_cast<f64>(c) - cx;
        for (usize r = 0; r < height; ++r) {
            const f64 dy = static_cast<f64>(r) - cy;
            const f64 v = keel::exp(-(dx * dx + dy * dy) * inv_2s2);
            out_col_major[c * height + r] = v;
            sum += v;
        }
    }
    if (sum > 0.0) {
        const f64 inv_sum = 1.0 / sum;
        for (usize i = 0; i < total; ++i) {
            out_col_major[i] *= inv_sum;
        }
    }
    return ImageStatus::kOk;
}

ImageStatus correlate_2d_replicate(ArrayRef<f64> img_col_major, usize height, usize width,
                                   ArrayRef<f64> kernel_col_major, usize kh, usize kw,
                                   MutArrayRef<f64> out_col_major) noexcept {
    if (height == 0 || width == 0) {
        return ImageStatus::kOk;
    }
    if (kh == 0 || kw == 0) {
        return ImageStatus::kInvalidArgument;
    }
    const usize total = height * width;
    if (img_col_major.size() < total || kernel_col_major.size() < kh * kw ||
        out_col_major.size() < total) {
        return ImageStatus::kInvalidArgument;
    }

    const i64 h_i = static_cast<i64>(height);
    const i64 w_i = static_cast<i64>(width);
    const i64 cy = static_cast<i64>(kh / 2);
    const i64 cx = static_cast<i64>(kw / 2);

    for (usize c = 0; c < width; ++c) {
        for (usize r = 0; r < height; ++r) {
            f64 acc = 0.0;
            for (usize kc = 0; kc < kw; ++kc) {
                i64 sc = static_cast<i64>(c) + static_cast<i64>(kc) - cx;
                if (sc < 0) sc = 0;
                else if (sc >= w_i) sc = w_i - 1;
                const usize sc_u = static_cast<usize>(sc);
                const usize k_col_off = kc * kh;
                const usize s_col_off = sc_u * height;
                for (usize kr = 0; kr < kh; ++kr) {
                    i64 sr = static_cast<i64>(r) + static_cast<i64>(kr) - cy;
                    if (sr < 0) sr = 0;
                    else if (sr >= h_i) sr = h_i - 1;
                    acc += img_col_major[s_col_off + static_cast<usize>(sr)] *
                           kernel_col_major[k_col_off + kr];
                }
            }
            out_col_major[c * height + r] = acc;
        }
    }
    return ImageStatus::kOk;
}

f64 ssim(ArrayRef<f64> a_col_major, ArrayRef<f64> b_col_major, usize height, usize width,
         f64 dynamic_range, Allocator* alloc) {
    if (height == 0 || width == 0) return 1.0;
    const usize total = height * width;
    if (a_col_major.size() < total || b_col_major.size() < total) {
        return Limits<f64>::quiet_nan();
    }

    f64 kernel[11 * 11]{};
    (void)gaussian_kernel_2d(11, 11, 1.5, kernel);

    ArenaTemp scratch = keel::get_scratch(alloc);
    Array<f64> mu_a(total, scratch.allocator());
    Array<f64> mu_b(total, scratch.allocator());
    Array<f64> aa(total, scratch.allocator());
    Array<f64> bb(total, scratch.allocator());
    Array<f64> ab(total, scratch.allocator());
    Array<f64> sa2(total, scratch.allocator());
    Array<f64> sb2(total, scratch.allocator());
    Array<f64> sab(total, scratch.allocator());

    (void)correlate_2d_replicate(a_col_major, height, width, kernel, 11, 11, mu_a);
    (void)correlate_2d_replicate(b_col_major, height, width, kernel, 11, 11, mu_b);

    for (usize i = 0; i < total; ++i) {
        const f64 a = a_col_major[i];
        const f64 b = b_col_major[i];
        aa[i] = a * a;
        bb[i] = b * b;
        ab[i] = a * b;
    }

    (void)correlate_2d_replicate(aa, height, width, kernel, 11, 11, sa2);
    (void)correlate_2d_replicate(bb, height, width, kernel, 11, 11, sb2);
    (void)correlate_2d_replicate(ab, height, width, kernel, 11, 11, sab);

    constexpr f64 k1 = 0.01;
    constexpr f64 k2 = 0.03;
    const f64 c1 = (k1 * dynamic_range) * (k1 * dynamic_range);
    const f64 c2 = (k2 * dynamic_range) * (k2 * dynamic_range);

    f64 sum = 0.0;
    usize cnt = 0;
    for (usize i = 0; i < total; ++i) {
        const f64 ma = mu_a[i];
        const f64 mb = mu_b[i];
        const f64 va = sa2[i] - ma * ma;
        const f64 vb = sb2[i] - mb * mb;
        const f64 cov = sab[i] - ma * mb;

        const f64 num = (2.0 * ma * mb + c1) * (2.0 * cov + c2);
        const f64 den = (ma * ma + mb * mb + c1) * (va + vb + c2);
        if (den > 0.0) {
            sum += num / den;
            ++cnt;
        }
    }
    return (cnt > 0) ? (sum / static_cast<f64>(cnt)) : 1.0;
}

} // namespace keel::image
