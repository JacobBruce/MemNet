#pragma once
#include <AudioFile.h>
#include "MathExt.h"

inline AudioFile<float>::AudioBuffer AverageSamples(const std::vector<float>& audio_buff, uint32_t avg_samples)
{
    uint32_t sampleCount = std::ceil(audio_buff.size() / (float)avg_samples);
    uint32_t buffIndex = 0;
    AudioFile<float>::AudioBuffer result;
    result.resize(1);
    result[0].resize(sampleCount);

    for (uint32_t i=0; i < sampleCount; ++i)
    {
        for (uint32_t l=0; l < avg_samples; ++l)
        {
            if (buffIndex < audio_buff.size()) {
                result[0][i] += audio_buff[buffIndex];
            } else {
                break;
            }
            buffIndex++;
        }

        result[0][i] /= avg_samples;
    }

    return result;
}

inline AudioFile<float> DownsampleAudio(const AudioFile<float>& audio, const float& sample_rate)
{
    double sampleIndex = 0.0;
    double sampleFrac = 0.0;
    double sampleStep = audio.getSampleRate() / sample_rate;
    double sampleRatio = sample_rate / audio.getSampleRate();
    uint32_t oldSampNum = audio.getNumSamplesPerChannel();
    uint32_t newSampNum = std::ceil(sampleRatio * oldSampNum);
    uint32_t avgSamples = std::floor(sampleStep);
    uint32_t sampleIndInt = 0;

    AudioFile<float>::AudioBuffer audioBuff;
    AudioFile<float> newAudio;

    if (avgSamples > 1) {
        audioBuff = AverageSamples(audio.samples[0], avgSamples);
    } else {
        audioBuff = audio.samples;
    }

    newAudio.setNumChannels(1);
    newAudio.setSampleRate(sample_rate);
    newAudio.setNumSamplesPerChannel(newSampNum);

    sampleStep = audioBuff[0].size() / (float)newSampNum;
    sampleIndex = sampleStep;

    if (sampleStep == 1.0 || std::abs(sampleStep-1.0) < 0.001) {
        newAudio.setAudioBuffer(audioBuff);
        return newAudio;
    }

    newAudio.samples[0][0] = audioBuff[0][0];

    for (uint32_t i=1; i < newSampNum; ++i)
    {
        sampleIndInt = std::floor(sampleIndex);
        sampleFrac = sampleIndex - sampleIndInt;

        if (sampleIndInt+1 >= audioBuff[0].size()) {
            newAudio.samples[0][i] = 0.f;
            break;
        }

        if (sampleFrac != 0.0) {
            newAudio.samples[0][i] = Lerp(sampleFrac, audioBuff[0][sampleIndInt], audioBuff[0][sampleIndInt+1]);
        } else {
            newAudio.samples[0][i] = audioBuff[0][sampleIndInt];
        }

        sampleIndex += sampleStep;
    }

    return newAudio;
}

inline bool NormalizeAudio(AudioFile<float>& audio, const float& min_avg_sample)
{
    double sampleSum = 0.0;
    uint32_t sampleNum = 0;

    for (uint32_t i=0; i < audio.getNumSamplesPerChannel(); ++i)
    {
        float sampleMag = std::abs(audio.samples[0][i]);

        if (sampleMag >= min_avg_sample) {
            sampleSum += sampleMag;
            sampleNum++;
        }
    }

    if (sampleNum == 0) return false;

    sampleSum = 4.0 * (sampleSum / sampleNum);

    for (uint32_t i=0; i < audio.getNumSamplesPerChannel(); ++i)
        audio.samples[0][i] = std::max(-1.f, std::min(1.f, float(audio.samples[0][i] / sampleSum)));

    return true;
}

inline void TrimAudio(AudioFile<float>& audio, const float& min_sample, const float& min_avg_sample)
{
    bool foundStart = false;
    uint32_t stopIndex = 0;
    uint32_t avgCount = 0;
    float avgSum = 0.f;
    float sampleMag = 0.f;

    std::vector<float> avgBuff(audio.getSampleRate(), 0.f);
    AudioFile<float>::AudioBuffer audioBuff;
    audioBuff.resize(1);
    audioBuff[0].reserve(avgBuff.size());

    for (int32_t i=audio.getNumSamplesPerChannel()-1; i >= 0; --i)
    {
        sampleMag = std::abs(audio.samples[0][i]);
        avgSum += sampleMag;

        if (sampleMag > min_sample) {
            stopIndex = i + avgCount;
            break;
        }

        if (++avgCount >= audio.getSampleRate()) {
            avgSum /= avgCount;

            if (avgSum > min_avg_sample) {
                stopIndex = i + audio.getSampleRate();
                break;
            }

            avgSum = 0.f;
            avgCount = 0;
        }

        if (i == 0) {
            if (avgCount > 0) {
                avgSum /= avgCount;

                if (avgSum > min_avg_sample) {
                    stopIndex = i + audio.getSampleRate();
                    break;
                }
            }

            return;
        }
    }

    avgSum = 0.f;
    avgCount = 0;

    for (uint32_t i=0; i < audio.getNumSamplesPerChannel(); ++i)
    {
        if (foundStart) {

            if (i >= stopIndex) break;
            audioBuff[0].emplace_back(audio.samples[0][i]);

        } else {

            avgBuff[avgCount] = audio.samples[0][i];
            sampleMag = std::abs(audio.samples[0][i]);
            avgSum += sampleMag;

            if (sampleMag > min_sample) {
                foundStart = true;
                for (uint32_t s=0; s < avgCount; ++s)
                    audioBuff[0].emplace_back(avgBuff[s]);
            }

            if (++avgCount >= audio.getSampleRate()) {
                avgSum /= avgCount;

                if (avgSum > min_avg_sample) {
                    foundStart = true;
                    for (const float& sample : avgBuff)
                        audioBuff[0].emplace_back(sample);
                }

                avgSum = 0.f;
                avgCount = 0;
            }
        }

        if (i == audio.getNumSamplesPerChannel()-1) {
            if (avgCount > 0) {
                avgSum /= avgCount;

                if (avgSum > min_avg_sample) {
                    foundStart = true;
                    for (uint32_t s=0; s < avgCount; ++s)
                        audioBuff[0].emplace_back(avgBuff[s]);
                }
            }
        }
    }

    if (foundStart)
        audio.setAudioBuffer(audioBuff);
}
