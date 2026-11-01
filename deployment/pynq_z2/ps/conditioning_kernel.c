#include <stdint.h>
#include <stddef.h>
#include <string.h>

#ifdef _WIN32
#define EXPORT __declspec(dllexport)
#else
#define EXPORT
#endif

EXPORT int conditioning_abi(void) { return 1; }

typedef struct {
    int channels, unity, fir, history;
    size_t stride;
    const double *gains;
    double *previous, *state;
    const double *kernels;
    const int32_t *delays;
    double *memory;
} conditioning_state;

EXPORT int conditioning_state_size(void) { return (int)sizeof(conditioning_state); }

EXPORT int conditioning_shift(const conditioning_state *plan, int samples) {
    double *memory = plan->memory;
    int channels = plan->channels, history = plan->history;
    size_t stride = plan->stride;
    if (channels < 1 || history < 0 || samples < 0 || stride < (size_t)history + samples) return 1;
    for (int channel = 0; channel < channels; ++channel)
        memmove(memory + (size_t)channel * stride, memory + (size_t)channel * stride + samples,
                (size_t)history * sizeof(double));
    return 0;
}

static void calibration(const double *memory, int samples, int channels, int history, size_t stride,
                        const double *kernels, const int32_t *delays, double *output) {
    for (int channel = 0; channel < channels; ++channel) {
        const double *signal = memory + (size_t)channel * stride + history - delays[channel] - 64;
        const double *kernel = kernels + (size_t)channel * 65;
        for (int block = 0; block < samples; block += 256) {
            double values[256] = {0};
            int count = samples - block < 256 ? samples - block : 256;
            for (int tap = 0; tap < 65; ++tap) {
                double coefficient = kernel[tap];
                for (int time = 0; time < count; ++time)
                    values[time] += coefficient * signal[block + time + tap];
            }
            for (int time = 0; time < count; ++time)
                output[(size_t)(block + time) * channels + channel] = values[time];
        }
    }
}

#define FEED(NAME, INPUT) \
EXPORT int NAME(const INPUT *input, int samples, const conditioning_state *plan, double *output) { \
    int channels = plan->channels, history = plan->history, unity = plan->unity, fir = plan->fir; \
    size_t stride = plan->stride; \
    const double *gains = plan->gains, *kernels = plan->kernels; \
    const int32_t *delays = plan->delays; \
    double *previous = plan->previous, *state = plan->state, *memory = plan->memory; \
    if (samples < 0 || channels < 1 || (fir && stride < (size_t)history + samples)) return 1; \
    if (fir) for (int channel = 0; channel < channels; ++channel) \
        if (delays[channel] < 0 || delays[channel] > history - 64) return 2; \
    for (int time = 0; time < samples; ++time) { \
        for (int channel = 0; channel < channels; ++channel) { \
            size_t index = (size_t)time * channels + channel; \
            double value = (double)input[index] / 32768.0; \
            state[channel] = value - previous[channel] + 0.995 * state[channel]; \
            previous[channel] = value; \
            double corrected = unity ? state[channel] : state[channel] / gains[channel]; \
            if (fir) memory[(size_t)channel * stride + history + time] = corrected; \
            else output[index] = corrected; \
        } \
    } \
    if (fir && samples) { \
        calibration(memory, samples, channels, history, stride, kernels, delays, output); \
        return conditioning_shift(plan, samples); \
    } \
    return 0; \
}

FEED(conditioning_pcm16, int16_t)
FEED(conditioning_float64, double)
