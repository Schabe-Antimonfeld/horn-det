#include <math.h>
#include <stddef.h>

#ifdef _WIN32
#define EXPORT __declspec(dllexport)
#else
#define EXPORT
#endif

EXPORT int stream_abi(void) { return 1; }

#define PHAT(NAME, REAL, SQRT, EPSILON) \
EXPORT int NAME(const REAL *spectrum, int frames, int channels, int bins, \
                const int *pairs, int count, REAL *output) { \
    if (frames < 1 || channels < 2 || bins < 2 || count < 1) return 1; \
    for (int frame = 0; frame < frames; ++frame) { \
        for (int pair = 0; pair < count; ++pair) { \
            int first = pairs[2 * pair], second = pairs[2 * pair + 1]; \
            if (first < 0 || second <= first || second >= channels) return 2; \
            const REAL *a = spectrum + 2 * ((size_t)frame * channels + first) * bins; \
            const REAL *b = spectrum + 2 * ((size_t)frame * channels + second) * bins; \
            REAL *target = output + 2 * ((size_t)frame * count + pair) * bins; \
            for (int bin = 0; bin < bins; ++bin) { \
                REAL real = a[2 * bin] * b[2 * bin] + a[2 * bin + 1] * b[2 * bin + 1]; \
                REAL imag = a[2 * bin + 1] * b[2 * bin] - a[2 * bin] * b[2 * bin + 1]; \
                REAL inverse = (REAL)1 / (SQRT(real * real + imag * imag) + EPSILON); \
                target[2 * bin] = real * inverse; \
                target[2 * bin + 1] = imag * inverse; \
            } \
        } \
    } \
    return 0; \
}

PHAT(stream_phat32, float, sqrtf, 1e-12f)
PHAT(stream_phat64, double, sqrt, 1e-12)

/* Preserve the reference's twelve-frame accumulation groups after cache reuse. */
#define REDUCE(NAME, REAL) \
EXPORT int NAME(const REAL *previous, int reused, const REAL *current, int fresh, \
                int width, double *output) { \
    int frames = reused + fresh; \
    if (reused < 0 || fresh < 1 || width < 1 || frames < 1) return 1; \
    for (int column = 0; column < width; ++column) output[column] = 0; \
    for (int start = 0; start < frames; start += 12) { \
        int stop = start + 12 < frames ? start + 12 : frames; \
        for (int column = 0; column < width; ++column) { \
            REAL total = 0; \
            for (int frame = start; frame < stop; ++frame) { \
                const REAL *source = frame < reused ? previous + (size_t)frame * width \
                                                   : current + (size_t)(frame - reused) * width; \
                total += source[column]; \
            } \
            output[column] += total; \
        } \
    } \
    return 0; \
}

REDUCE(stream_reduce32, float)
REDUCE(stream_reduce64, double)

/* Cache sums in the next window's twelve-frame groups, including its partial group. */
#define FUSED(NAME, REAL, SQRT, EPSILON) \
EXPORT int NAME(const REAL *spectrum, int frames, int channels, int bins, const int *pairs, int count, \
                const REAL *previous, int reused, int retained, double *output, REAL *future, REAL *partial) { \
    if (frames < 1 || channels < 2 || bins < 2 || count < 1 || reused < 0 || retained < 0 || retained > frames) return 1; \
    size_t width = (size_t)count * bins * 2; \
    for (size_t column = 0; column < width; ++column) { \
        double total = 0; \
        for (int group = 0; group < reused / 12; ++group) total += previous[(size_t)group * width + column]; \
        output[column] = total; \
        partial[column] = reused % 12 ? previous[(size_t)(reused / 12) * width + column] : 0; \
    } \
    for (size_t column = 0; column < ((retained + 11) / 12) * width; ++column) future[column] = 0; \
    for (int frame = 0; frame < frames; ++frame) { \
        int flush = (reused + frame + 1) % 12 == 0 || frame + 1 == frames; \
        int next = frame - (frames - retained); \
        for (int pair = 0; pair < count; ++pair) { \
            int first = pairs[2 * pair], second = pairs[2 * pair + 1]; \
            if (first < 0 || second <= first || second >= channels) return 2; \
            const REAL *a = spectrum + 2 * ((size_t)frame * channels + first) * bins; \
            const REAL *b = spectrum + 2 * ((size_t)frame * channels + second) * bins; \
            for (int bin = 0; bin < bins; ++bin) { \
                REAL real = a[2 * bin] * b[2 * bin] + a[2 * bin + 1] * b[2 * bin + 1]; \
                REAL imag = a[2 * bin + 1] * b[2 * bin] - a[2 * bin] * b[2 * bin + 1]; \
                REAL inverse = (REAL)1 / (SQRT(real * real + imag * imag) + EPSILON); \
                size_t column = 2 * ((size_t)pair * bins + bin); \
                REAL r = real * inverse, i = imag * inverse; \
                partial[column] += r; partial[column + 1] += i; \
                if (next >= 0) { \
                    size_t target = (size_t)(next / 12) * width + column; \
                    future[target] += r; future[target + 1] += i; \
                } \
                if (flush) { \
                    output[column] += partial[column]; output[column + 1] += partial[column + 1]; \
                    partial[column] = 0; partial[column + 1] = 0; \
                } \
            } \
        } \
    } \
    return 0; \
}

FUSED(stream_fused32, float, sqrtf, 1e-12f)
FUSED(stream_fused64, double, sqrt, 1e-12)

EXPORT int stream_dc(const double *audio, int samples, int channels, const double *gains,
                     double *previous, double *state, double *output) {
    if (samples < 0 || channels < 1) return 1;
    for (int time = 0; time < samples; ++time) {
        for (int channel = 0; channel < channels; ++channel) {
            size_t index = (size_t)time * channels + channel;
            double value = audio[index] / 32768.0;
            state[channel] = value - previous[channel] + 0.995 * state[channel];
            previous[channel] = value;
            output[index] = state[channel] / gains[channel];
        }
    }
    return 0;
}
