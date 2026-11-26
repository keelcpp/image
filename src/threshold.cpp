#include "keel/image.h"

#include "keel/arena.h"
#include "keel/array.h"
#include "keel/math.h"

namespace keel::image {

ImageStatus histogram(ArrayRef<f64> img, f64 min_val, f64 max_val, MutArrayRef<f64> counts) noexcept {
    const usize nbins = counts.size();
    if (nbins == 0 || !keel::is_finite(min_val) || !keel::is_finite(max_val) ||
        !(max_val >= min_val)) {
        return ImageStatus::kInvalidArgument;
    }
    for (usize i = 0; i < nbins; ++i) {
        counts[i] = 0.0;
    }
    const f64 span = max_val - min_val;
    const f64 scale = (nbins > 1 && span > 0.0) ? (as<f64>(nbins - 1) / span) : 0.0;
    for (usize i = 0; i < img.size(); ++i) {
        const f64 v = img[i];
        if (keel::is_nan(v)) continue;
        usize bin = 0;
        if (nbins > 1 && span > 0.0) {
            const f64 pos = keel::round((v - min_val) * scale);
            if (pos <= 0.0) {
                bin = 0;
            } else if (pos >= as<f64>(nbins - 1)) {
                bin = nbins - 1;
            } else {
                bin = as<usize>(pos);
            }
        }
        counts[bin] += 1.0;
    }
    return ImageStatus::kOk;
}

OtsuResult otsu_threshold(ArrayRef<f64> hist_counts, Allocator* alloc) {
    const usize l = hist_counts.size();
    OtsuResult res{};
    if (l == 0) return res;

    f64 total = 0.0;
    f64 sum_total = 0.0;
    for (usize i = 0; i < l; ++i) {
        const f64 c = hist_counts[i];
        if (!(c >= 0.0) || !keel::is_finite(c)) return res;
        total += c;
        sum_total += as<f64>(i) * c;
    }
    if (total <= 0.0 || l == 1) {
        return res;
    }

    ArenaTemp scratch = keel::get_scratch(alloc);
    Array<f64> sigma_b(l, 0.0, scratch.allocator());

    f64 w0 = 0.0;
    f64 sum0 = 0.0;
    for (usize t = 0; t + 1 < l; ++t) {
        w0 += hist_counts[t];
        sum0 += as<f64>(t) * hist_counts[t];
        const f64 w1 = total - w0;
        if (w0 == 0.0 || w1 == 0.0) continue;
        const f64 mu0 = sum0 / w0;
        const f64 mu1 = (sum_total - sum0) / w1;
        const f64 diff = mu0 - mu1;
        sigma_b[t] = w0 * w1 * diff * diff;
    }

    f64 best_var = sigma_b[0];
    for (usize t = 1; t + 1 < l; ++t) {
        if (sigma_b[t] > best_var) best_var = sigma_b[t];
    }

    usize n_tied = 0;
    f64 sum_tied = 0.0;
    for (usize t = 0; t + 1 < l; ++t) {
        if (sigma_b[t] == best_var) {
            sum_tied += as<f64>(t);
            ++n_tied;
        }
    }
    res.bin_level = (n_tied > 0) ? (sum_tied / as<f64>(n_tied)) : 0.0;
    res.threshold = res.bin_level / as<f64>(l - 1);

    const f64 mu = sum_total / total;
    f64 total_var = 0.0;
    for (usize i = 0; i < l; ++i) {
        const f64 d = as<f64>(i) - mu;
        total_var += hist_counts[i] * d * d;
    }
    res.effectiveness = (total_var > 0.0) ? (best_var / (total * total_var)) : 0.0;
    return res;
}

} // namespace keel::image
