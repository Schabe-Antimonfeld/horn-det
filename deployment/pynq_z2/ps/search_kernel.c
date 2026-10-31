#include <stdint.h>
#include <math.h>
#include <string.h>

#ifdef _WIN32
#define EXPORT __declspec(dllexport)
#else
#define EXPORT __attribute__((visibility("default")))
#endif

EXPORT int srp_abi(void) {
    return 2;
}

EXPORT int srp_phat(const double * restrict spectrum, int frames, int channels, int bins,
                    const int32_t * restrict pairs, int count, double * restrict output) {
    if (frames < 1 || channels < 2 || bins < 2 || count < 1) return -1;
    memset(output, 0, sizeof(double) * 2 * count * bins);
    for (int frame = 0; frame < frames; frame++) {
        for (int pair = 0; pair < count; pair++) {
            const double *first = spectrum + 2 * ((frame * channels + pairs[2 * pair]) * bins);
            const double *second = spectrum + 2 * ((frame * channels + pairs[2 * pair + 1]) * bins);
            double *target = output + 2 * pair * bins;
            #pragma GCC ivdep
            for (int bin = 0; bin < bins; bin++) {
                const double ar = first[2 * bin], ai = first[2 * bin + 1];
                const double br = second[2 * bin], bi = second[2 * bin + 1];
                const double real = ar * br + ai * bi;
                const double imag = ai * br - ar * bi;
                const double magnitude = sqrt(real * real + imag * imag);
                const double inverse = 1.0 / (magnitude + 1e-12);
                target[2 * bin] += real * inverse;
                target[2 * bin + 1] += imag * inverse;
            }
        }
    }
    return 0;
}

EXPORT int srp_phat32(const float * restrict spectrum, int frames, int channels, int bins,
                      const int32_t * restrict pairs, int count, float * restrict output) {
    if (frames < 1 || channels < 2 || bins < 2 || count < 1) return -1;
    memset(output, 0, sizeof(float) * 2 * count * bins);
    for (int frame = 0; frame < frames; frame++) {
        for (int pair = 0; pair < count; pair++) {
            const float *first = spectrum + 2 * ((frame * channels + pairs[2 * pair]) * bins);
            const float *second = spectrum + 2 * ((frame * channels + pairs[2 * pair + 1]) * bins);
            float *target = output + 2 * pair * bins;
            #pragma GCC ivdep
            for (int bin = 0; bin < bins; bin++) {
                const float ar = first[2 * bin], ai = first[2 * bin + 1];
                const float br = second[2 * bin], bi = second[2 * bin + 1];
                const float real = ar * br + ai * bi;
                const float imag = ai * br - ar * bi;
                const float inverse = 1.0f / (sqrtf(real * real + imag * imag) + 1e-12f);
                target[2 * bin] += real * inverse;
                target[2 * bin + 1] += imag * inverse;
            }
        }
    }
    return 0;
}

static double direction_score(const double *correlations, const int32_t *first,
                              const int32_t *second, const double *fraction,
                              int pairs, int direction) {
    double totals[4] = {0, 0, 0, 0};
    const int offset = direction * pairs;
    int pair = 0;
    for (; pair + 3 < pairs; pair += 4) {
        for (int lane = 0; lane < 4; lane++) {
            const int index = offset + pair + lane;
            const double lower = correlations[first[index]];
            totals[lane] += lower + fraction[index] * (correlations[second[index]] - lower);
        }
    }
    double total = (totals[0] + totals[1]) + (totals[2] + totals[3]);
    for (; pair < pairs; pair++) {
        const int index = offset + pair;
        const double lower = correlations[first[index]];
        total += lower + fraction[index] * (correlations[second[index]] - lower);
    }
    return total;
}

EXPORT int srp_scores(const double *correlations, const int32_t *first,
                      const int32_t *second, const double *fraction, int pairs,
                      const int32_t *directions, int count, double *output) {
    if (pairs < 1 || count < 0) return -1;
    for (int index = 0; index < count; index++) {
        output[index] = direction_score(correlations, first, second, fraction, pairs, directions[index]);
    }
    return 0;
}

EXPORT int srp_refine(const double *correlations, const int32_t *first,
                      const int32_t *second, const double *fraction, int pairs,
                      const int32_t *offsets, const int32_t *directions,
                      const int32_t *anchors, const int32_t *cells, int count,
                      const double *bounds, int pruning, double *scores, int *rejected) {
    if (count < 1 || count > 25 || pairs < 1) return -1;
    unsigned char visited[25] = {0};
    int winner = anchors[cells[0]];
    *rejected = 0;
    for (int index = 1; index < count; index++) {
        const int anchor = anchors[cells[index]];
        if (scores[anchor] > scores[winner]) winner = anchor;
    }
    for (;;) {
        int chosen = -1;
        for (int index = 0; index < count; index++) {
            if (visited[index]) continue;
            if (pruning && bounds[cells[index]] < scores[winner]) {
                visited[index] = 1;
                (*rejected)++;
            } else if (chosen < 0 || bounds[cells[index]] > bounds[cells[chosen]]) {
                chosen = index;
            }
        }
        if (chosen < 0) break;
        visited[chosen] = 1;
        const int cell = cells[chosen];
        for (int index = offsets[cell]; index < offsets[cell + 1]; index++) {
            const int direction = directions[index];
            if (isnan(scores[direction])) {
                scores[direction] = direction_score(correlations, first, second, fraction, pairs, direction);
            }
            if (scores[direction] > scores[winner]) winner = direction;
        }
    }
    return winner;
}
