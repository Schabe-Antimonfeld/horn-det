#include <math.h>
#include <stddef.h>

#ifdef _WIN32
#define API __declspec(dllexport)
#else
#define API __attribute__((visibility("default")))
#endif

API int dsp_abi(void) { return 3; }

API int dsp_prepare(const double *audio, int samples, int channels, const double *window,
                    int fold, int first, double *output) {
    if (!audio || !window || !output || samples < 1024 || channels < 1 || channels > 64 ||
        (fold != 1 && fold != 2 && fold != 4 && fold != 8) || first < 0 || first > (samples-1024)/512) return 1;
    for (size_t i = 0; i < (size_t)samples * channels; ++i) if (!isfinite(audio[i])) return 2;
    for (int i = 0; i < 1024; ++i) if (!isfinite(window[i])) return 2;
    int length = 1024 / fold, count = 1 + (samples-1024)/512 - first;
    for (int frame = 0; frame < count; ++frame) {
        const double *source = audio + (size_t)(frame+first) * 512 * channels;
        double *target = output + (size_t)frame * channels * length;
        for (int bin = 0; bin < length; ++bin) {
            if (fold == 1) {
                for (int channel = 0; channel < channels; ++channel)
                    target[channel*length+bin] = source[(size_t)bin*channels+channel] * window[bin];
            } else {
                double total[64] = {0};
                for (int part = 0; part < fold; ++part) {
                    int sample = bin + part * length;
                    for (int channel = 0; channel < channels; ++channel)
                        total[channel] += source[(size_t)sample*channels+channel] * window[sample];
                }
                for (int channel = 0; channel < channels; ++channel) target[channel*length+bin] = total[channel];
            }
        }
    }
    return 0;
}

API int dsp_beam(const double *input, const double *weights, size_t blocks,
                 size_t channels, size_t bins, ptrdiff_t block_stride,
                 ptrdiff_t channel_stride, ptrdiff_t frequency_stride,
                 ptrdiff_t weight_channel, ptrdiff_t weight_frequency, double *output) {
    if (!input || !weights || !output || !blocks || !channels || !bins) return 1;
    for (size_t b = 0; b < blocks; ++b) {
        double *out = output + b * bins * 2;
        for (size_t f = 0; f < bins; ++f) {
            double real = 0, imaginary = 0;
            for (size_t c = 0; c < channels; ++c) {
                const double *x = input + (ptrdiff_t)b * block_stride + (ptrdiff_t)c * channel_stride + (ptrdiff_t)f * frequency_stride;
                const double *w = weights + (ptrdiff_t)c * weight_channel + (ptrdiff_t)f * weight_frequency;
                real += x[0] * w[0] - x[1] * w[1];
                imaginary += x[0] * w[1] + x[1] * w[0];
            }
            out[2*f] = real;
            out[2*f+1] = imaginary;
        }
    }
    return 0;
}

API int dsp_mel(const double *input, size_t frames, const double *weights,
                const int *starts, const int *ends, double *output) {
    if (!input || !weights || !starts || !ends || !output || !frames) return 1;
    for (size_t t = 0; t < frames; ++t) {
        double power[513];
        for (size_t f = 0; f < 513; ++f) {
            double real = input[(t*513+f)*2], imaginary = input[(t*513+f)*2+1];
            power[f] = real * real + imaginary * imaginary;
        }
        for (size_t m = 0; m < 40; ++m) {
            if (starts[m] < 0 || ends[m] > 513 || starts[m] > ends[m]) return 2;
            double first = 0, second = 0, third = 0, fourth = 0;
            int f = starts[m];
            for (; f + 3 < ends[m]; f += 4) {
                first += power[f] * weights[m*513+f];
                second += power[f+1] * weights[m*513+f+1];
                third += power[f+2] * weights[m*513+f+2];
                fourth += power[f+3] * weights[m*513+f+3];
            }
            double total = (first + second) + (third + fourth);
            for (; f < ends[m]; ++f) total += power[f] * weights[m*513+f];
            output[t*40+m] = log(total + 1e-12);
        }
    }
    return 0;
}

API int dsp_temporal(const double *input, double *output) {
    if (!input || !output) return 1;
    double energy[7], average = 0, stats[7][160];
    for (int c = 0; c < 7; ++c) {
        energy[c] = 0;
        for (int i = 0; i < 960; ++i) energy[c] += input[c*960+i];
        energy[c] /= 960;
        average += energy[c];
    }
    average /= 7;
    for (int c = 0; c < 7; ++c) {
        for (int m = 0; m < 40; ++m) {
            double mean = 0, variance = 0, delta = 0;
            double values[24], low = 0, high = 0;
            for (int t = 0; t < 24; ++t) {
                values[t] = input[c*960+m*24+t] - energy[c];
                mean += values[t];
                if (t == 0 || values[t] < low) low = values[t];
                if (t == 0 || values[t] > high) high = values[t];
                if (t) delta += fabs(values[t] - values[t-1]);
            }
            mean /= 24;
            for (int t = 0; t < 24; ++t) {
                double difference = values[t] - mean;
                variance += difference * difference;
            }
            stats[c][m] = mean + energy[c] - average;
            stats[c][40+m] = sqrt(variance / 24);
            stats[c][80+m] = delta / 23;
            stats[c][120+m] = high - low;
        }
    }
    const int starts[3] = {0, 3, 5}, ends[3] = {3, 5, 7};
    for (int i = 0; i < 160; ++i) {
        for (int group = 0; group < 3; ++group) {
            double sum = 0;
            for (int c = starts[group]; c < ends[group]; ++c) sum += stats[c][i];
            output[group*160+i] = sum / (ends[group] - starts[group]);
        }
        double peak = stats[0][i];
        for (int c = 1; c < 7; ++c) if (stats[c][i] > peak) peak = stats[c][i];
        output[480+i] = peak;
    }
    return 0;
}
