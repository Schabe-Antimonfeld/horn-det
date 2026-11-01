#include <stddef.h>

#ifdef _WIN32
#define EXPORT __declspec(dllexport)
#else
#define EXPORT
#endif

EXPORT int refinement_abi(void) { return 1; }

EXPORT int refinement_prepare(const double *audio, int samples, int channels,
                             const double *window, int fold, int first,
                             int count, double *output) {
    if (!audio || !window || !output || samples < 1024 || channels < 1 || channels > 64 ||
        (fold != 1 && fold != 2 && fold != 4 && fold != 8) || first < 0 || count < 1 ||
        first > (samples - 1024) / 512 || count > 1 + (samples - 1024) / 512 - first)
        return -1;
    const int length = 1024 / fold;
    for (int frame = 0; frame < count; ++frame) {
        for (int channel = 0; channel < channels; ++channel) {
            double *target = output + ((size_t)frame * channels + channel) * length;
            const double *source = audio + (size_t)(first + frame) * 512 * channels + channel;
            for (int bin = 0; bin < length; ++bin) {
                double total = 0.0;
                if (fold == 1) {
                    target[bin] = source[(size_t)bin * channels] * window[bin];
                    continue;
                }
                for (int part = 0; part < fold; ++part) {
                    int sample = bin + part * length;
                    total += source[(size_t)sample * channels] * window[sample];
                }
                target[bin] = total;
            }
        }
    }
    return 0;
}
