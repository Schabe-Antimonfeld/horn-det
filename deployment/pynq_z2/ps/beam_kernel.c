#include <stddef.h>

#ifdef _WIN32
#define EXPORT __declspec(dllexport)
#else
#define EXPORT
#endif

EXPORT int beam_abi(void) { return 1; }

/* Input is channel-major; kernels and shifts are beam-major then channel-major. */
EXPORT int beam_fir(const double *restrict audio, int samples, int channels,
                    const double *restrict kernels, const int *restrict shifts,
                    int beams, double *restrict output) {
    if (!audio || !kernels || !shifts || !output || samples < 1 || channels < 1 || beams < 1)
        return 1;
    for (int beam = 0; beam < beams; ++beam) {
        double *result = output + (size_t)beam * samples;
        for (int time = 0; time < samples; ++time) result[time] = 0.0;
        for (int channel = 0; channel < channels; ++channel) {
            int integer = shifts[(size_t)beam * channels + channel];
            if (integer < 32) return 2;
            const double *signal = audio + (size_t)channel * samples;
            const double *kernel = kernels + ((size_t)beam * channels + channel) * 65;
            for (int block = integer; block < samples; block += 256) {
                double values[256] = {0.0};
                int limit = samples - block < 256 ? samples : block + 256;
                for (int tap = 0; tap < 65; ++tap) {
                    int offset = 32 - integer - tap;
                    int first = block < -offset ? -offset : block;
                    int last = limit > samples - offset ? samples - offset : limit;
                    double coefficient = kernel[tap];
                    for (int time = first; time < last; ++time)
                        values[time - block] += coefficient * signal[time + offset];
                }
                for (int time = block; time < limit; ++time) result[time] += values[time - block];
            }
        }
        for (int time = 0; time < samples; ++time) result[time] /= channels;
    }
    return 0;
}
