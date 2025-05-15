#pragma once
#include <cstdlib>
#include <string>
#include <iostream>
#include <fstream>
#include <algorithm>
#include <thread>
#include <atomic>
#include "ReadWrite.h"
#include "AudioUtils.h"
#include "Worderizer.h"
#include "MemNet.h"
#include "Timer.h"

using namespace std::chrono_literals;

inline void CheckTextFiles(const std::string& data_dir,
                    phmap::parallel_flat_hash_map<std::u32string,uint32_t>& word_map,
                    float min_byte_token_ratio=2.0f)
{
    std::vector<uint32_t> tokens;
    std::vector<std::string> textFiles = ListAllFiles(data_dir);

    if (word_map.empty())
        Worderizer::LoadWordMap(word_map, GLOBALS::config_map["TOKENIZER"]);

    for (size_t i=0; i < textFiles.size(); ++i)
    {
        std::string textStr = ReadFileStr(textFiles[i]);
        tokens.clear();

        if (Worderizer::StrToTokens(Worderizer::U8ToU32(textStr), tokens, word_map)) {
            float byteRatio = (float)textStr.size() / tokens.size();
            if (byteRatio < min_byte_token_ratio) {
                std::cout << "Outlier file: " << textFiles[i] << std::endl;
                std::cout << "bytes/tokens ratio: " << byteRatio << std::endl;
            }
        } else {
            HandleFatalError("Failed to tokenize text in "+textFiles[i]);
        }
    }
}

inline std::function<void(const std::string&, const uint32_t&)> GetDataFunc(const std::string& func,
                    const std::string& func_arg, std::vector<std::vector<float>>& dest, uint32_t array_size)
{
    std::function<void(const std::string&, const uint32_t&)> result;

    try {
        if (func == "INDEX_TO") {

            float posVal, negVal;

            if (func_arg == "ARRAY") {
                posVal = GLOBALS::POS_LABEL;
                negVal = GLOBALS::NEG_LABEL;
            } else {
                std::vector<std::string> argParts(ExplodeStr(func_arg, "_"));
                posVal = stof(argParts[0]);
                negVal = stof(argParts[1]);
            }

            result = [=, &dest] (const std::string& v, const uint32_t& i) {

                uint32_t setIndex = stoul(v);

                for (uint32_t l=0; l < array_size; ++l)
                {
                    if (l == setIndex) {
                        dest[i].emplace_back(posVal);
                    } else {
                        dest[i].emplace_back(negVal);
                    }
                }
            };

        } else if (func == "DIV") {

            float divArg = stof(func_arg);

            result = [=, &dest] (const std::string& v, const uint32_t& i) {
                dest[i].emplace_back(stof(v)/divArg);
            };

        } else if (func == "MUL") {

            float mulArg = stof(func_arg);

            result = [=, &dest] (const std::string& v, const uint32_t& i) {
                dest[i].emplace_back(stof(v)*mulArg);
            };

        } else if (func == "ADD") {

            float addArg = stof(func_arg);

            result = [=, &dest] (const std::string& v, const uint32_t& i) {
                dest[i].emplace_back(stof(v)+addArg);
            };

        } else {

            result = [&] (const std::string& v, const uint32_t& i) {
                dest[i].emplace_back(stof(v));
            };
        }

    } catch (...) {
        HandleFatalError("invalid config setting: "+func+"_"+func_arg);
    }

    return result;
}

inline void CalcInOutData(uint64_t& expected_ins, uint64_t& expected_outs, std::string& in_func,
                          std::string& out_func, std::string& in_func_arg, std::string& out_func_arg)
{
    if (StrStartsWith(in_func, "INDEX_TO_")) {
        in_func_arg = ReplaceStr(const_cast<const std::string&>(in_func), "INDEX_TO_", "");
        in_func = "INDEX_TO";
        expected_ins = 1;
    } else if (StrStartsWith(in_func, "DIV_") || StrStartsWith(in_func, "MUL_") || StrStartsWith(in_func, "DIV_")) {
        std::vector<std::string> strParts(ExplodeStr(in_func, "_"));
        in_func = strParts[0];
        in_func_arg = strParts[1];
    }

    if (StrStartsWith(out_func, "INDEX_TO_")) {
        out_func_arg = ReplaceStr(const_cast<const std::string&>(out_func), "INDEX_TO_", "");
        out_func = "INDEX_TO";
        expected_outs = 1;
    } else if (StrStartsWith(out_func, "ADD_") || StrStartsWith(out_func, "MUL_") || StrStartsWith(out_func, "DIV_")) {
        std::vector<std::string> strParts(ExplodeStr(out_func, "_"));
        out_func = strParts[0];
        out_func_arg = strParts[1];
    }
}

inline bool IsHeadOfCSV(const std::string& line)
{
    for (const auto& c : line)
        if (std::isalpha(c)) return true;

    return false;
}

inline uint32_t LoadDataFromCSV(const std::string& data_file, std::vector<std::vector<float>>& batches, uint64_t& batch_size, uint64_t& expected_vals,
                                uint32_t& examples_per_batch, std::function<void(const std::string&, const uint32_t&)>& process_data_func)
{
    uint32_t exampleCount = 0;
    std::ifstream ifs(data_file);
    std::string line;

    if (ifs.is_open()) {

        uint64_t batchIndex = batches.size();

        if (batches.empty()) {
            batches.emplace_back();
            batches[batchIndex].reserve(batch_size);
        } else {
            batchIndex--;
        }

        uint64_t exampleCounter = batches[batchIndex].size() / expected_vals;

        while (!ifs.eof())
        {
            std::getline(ifs, line);
            std::vector<std::string> vals(ExplodeStr(line, ","));
            vals = ExplodeStr(line, ",");

            if (line == "") break;

            if (vals.size() != expected_vals) {
                std::stringstream errSS;
                errSS << "data file doesn't fit to net inputs" << std::endl;
                errSS << "expected values per line: " << expected_vals << std::endl;
                errSS << "actual values on line " << (exampleCounter+1) << ": " << vals.size() << std::endl;
                HandleFatalError(errSS.str());
            }

            if (vals.size() != expected_vals)
                HandleFatalError("csv file is corrupted!");

            if (exampleCounter++ == 0 && IsHeadOfCSV(line)) {
                exampleCounter = 0;
                continue;
            }

            try {
                for (uint32_t i=0; i < expected_vals; ++i)
                    process_data_func(vals[i], batchIndex);
                exampleCount++;
            } catch (...) {
                if (exampleCounter == 1) {
                    exampleCounter = 0;
                } else {
                    HandleFatalError("csv file is corrupted!");
                }
            }

            if (exampleCounter >= examples_per_batch) {
                batchIndex++;
                exampleCounter = 0;
                batches.emplace_back();
                batches[batchIndex].reserve(batch_size);
            }
        }

        ifs.close();

    } else {
        std::cout << "WARNING: failed to open " << data_file << std::endl;
    }

    return exampleCount;
}

inline void LoadDataFiles(std::string& input_file, std::string& output_file,
                          std::string& input_func, std::string& output_func,
                          uint32_t& examples_per_batch, MemNet& net,
                          std::vector<std::vector<float>>& in_batches,
                          std::vector<std::vector<float>>& out_batches)
{
    static bool calcInOutData = true;
    static uint64_t inBatchSize = net.InputSize() * examples_per_batch;
    static uint64_t outBatchSize = net.OutputSize() * examples_per_batch;
    static uint64_t expectedIns = net.InputSize();
    static uint64_t expectedOuts = net.OutputSize();
    static std::string inFunc = input_func;
    static std::string outFunc = output_func;
    static std::string inFuncArg, outFuncArg;

    static std::function<void(const std::string&, const uint32_t&)> processInData;
    static std::function<void(const std::string&, const uint32_t&)> processOutData;

    if (calcInOutData) {
        CalcInOutData(expectedIns, expectedOuts, inFunc, outFunc, inFuncArg, outFuncArg);
        processInData = GetDataFunc(inFunc, inFuncArg, in_batches, net.InputSize());
        processOutData = GetDataFunc(outFunc, outFuncArg, out_batches, net.OutputSize());
        calcInOutData = false;
    }

    uint32_t inExamples = LoadDataFromCSV(input_file, in_batches, inBatchSize, expectedIns, examples_per_batch, processInData);
    uint32_t outExamples = LoadDataFromCSV(output_file, out_batches, outBatchSize, expectedOuts, examples_per_batch, processOutData);

    if (inExamples != outExamples)
        HandleFatalError("number of examples in " + input_file + " doesn't match number in " + output_file);
}

inline void LoadDataBatches(std::string& data_file, std::string& data_type, std::string& input_func,
                     std::string& output_func, uint32_t& examples_per_batch, MemNet& net,
                     std::vector<std::vector<float>>& in_batches,
                     std::vector<std::vector<float>>& out_batches,
                     std::vector<std::vector<uint32_t>>& token_batches,
                     phmap::parallel_flat_hash_map<std::u32string,uint32_t>& word_map)
{
    uint64_t inBatchSize = net.InputSize() * examples_per_batch;
    uint64_t outBatchSize = net.OutputSize() * examples_per_batch;
    uint64_t expectedIns = net.InputSize();
    uint64_t expectedOuts = net.OutputSize();
    std::string inFunc(input_func);
    std::string outFunc(output_func);
    std::string inFuncArg, outFuncArg;

    std::cout << "Loading data from " << data_file << std::endl;

    CalcInOutData(expectedIns, expectedOuts, inFunc, outFunc, inFuncArg, outFuncArg);

    std::function<void(const std::string&, const uint32_t&)> processInData(GetDataFunc(inFunc, inFuncArg, in_batches, net.InputSize()));
    std::function<void(const std::string&, const uint32_t&)> processOutData(GetDataFunc(outFunc, outFuncArg, out_batches, net.OutputSize()));

    in_batches.clear();
    out_batches.clear();
    token_batches.clear();

    if (data_type == "CSV_IN") {

        LoadDataFromCSV(data_file, in_batches, inBatchSize, expectedIns, examples_per_batch, processInData);

        if (in_batches.back().empty()) in_batches.pop_back();

    } else if (data_type == "CSV_OUT_IN" || data_type == "CSV_IN_OUT") {

        std::ifstream ifs(data_file);
        std::string line;

        if (ifs.is_open()) {

            uint64_t expectedVals = expectedIns + expectedOuts;
            uint64_t inputOffset = expectedOuts;
            uint64_t outputOffset = 0;
            uint64_t exampleCounter = 0;
            uint64_t batchIndex = 0;

            if (data_type == "CSV_IN_OUT") {
                inputOffset = 0;
                outputOffset = expectedIns;
            }

            in_batches.emplace_back();
            out_batches.emplace_back();
            in_batches[0].reserve(inBatchSize);
            out_batches[0].reserve(outBatchSize);

            while (!ifs.eof())
            {
                std::getline(ifs, line);
                std::vector<std::string> vals(ExplodeStr(line, ","));
                vals = ExplodeStr(line, ",");

                if (line == "") break;

                if (vals.size() != expectedVals) {
                    std::stringstream errSS;
                    errSS << "data file doesn't fit to net inputs/outputs" << std::endl;
                    errSS << "expected values per line: " << expectedVals << std::endl;
                    errSS << "actual values per line: " << vals.size() << std::endl;
                    HandleFatalError(errSS.str());
                }

                if (vals.size() != expectedVals)
                    HandleFatalError("csv file is corrupted!");

                if (exampleCounter++ == 0 && IsHeadOfCSV(line)) {
                    exampleCounter = 0;
                    continue;
                }

                try {
                    for (uint32_t i=0; i < expectedIns; ++i)
                        processInData(vals[inputOffset+i], batchIndex);

                    for (uint32_t i=0; i < expectedOuts; ++i)
                        processOutData(vals[outputOffset+i], batchIndex);
                } catch (...) {
                    if (exampleCounter == 1) {
                        exampleCounter = 0;
                    } else {
                        HandleFatalError("csv file is corrupted!");
                    }
                }

                if (exampleCounter >= examples_per_batch) {
                    batchIndex++;
                    exampleCounter = 0;
                    in_batches.emplace_back();
                    out_batches.emplace_back();
                    in_batches[batchIndex].reserve(inBatchSize);
                    out_batches[batchIndex].reserve(outBatchSize);
                }
            }

            if (in_batches.back().empty()) {
                in_batches.pop_back();
                out_batches.pop_back();
            }

            ifs.close();
        } else {
            HandleFatalError("failed to open file: "+data_file);
        }

    } else if (data_type == "AUDIO_FILES") {

        std::vector<std::string> audioFiles(ListAllFiles(data_file));
        const float sampleRate = stof(GLOBALS::config_map["SAMPLE_RATE"]);
        const float minSample = stof(GLOBALS::config_map["MIN_SAMPLE"]);
        const float minAvgSample = stof(GLOBALS::config_map["MIN_AVG_SAMPLE"]);
        const float trimAudio = StrToUpper(GLOBALS::config_map["TRIM_AUDIO"]) == "TRUE";

        uint32_t exampleCounter = 0;
        uint64_t batchIndex = 0;

        in_batches.emplace_back();
        in_batches[0].reserve(inBatchSize);

        for (const std::string& audioFile : audioFiles)
        {
            AudioFile<float> audio;
            uint32_t sampleCounter = 0;

            if (StrEndsWith(audioFile, ".wav") || StrEndsWith(audioFile, ".WAV")) {
                if (!audio.load(audioFile)) {
                    std::cout << "WARNING: failed to read audio from " << audioFile;
                    continue;
                }
            } else {
                //TODO: mp3 support
                std::cout << "WARNING: skipping unsupported file: " << audioFile;
                continue;
            }

            if (audio.getSampleRate() <= 0) {
                std::cout << "WARNING: corrupted audio file detected: " << audioFile << std::endl;
                continue;
            } else if (audio.getSampleRate() < sampleRate) {
                std::cout << "WARNING: faudio sample rate is too low: " << audioFile << std::endl;
                continue;
            } else if (audio.getSampleRate() != sampleRate) {
                //std::cout << "Downsampling " << audioFile << std::endl;
                audio = DownsampleAudio(audio, sampleRate);
            }

            if (trimAudio) TrimAudio(audio, minSample, minAvgSample);

            if (!NormalizeAudio(audio, minAvgSample)) {
                std::cout << "WARNING: detected silent audio file: " << audioFile;
                continue;
            }

            for (uint32_t i=0; i < audio.getNumSamplesPerChannel(); ++i)
            {
                in_batches[batchIndex].emplace_back(audio.samples[0][i]);

                if (++sampleCounter >= net.InputSize()) {
                    if (++exampleCounter >= examples_per_batch) {
                        batchIndex++;
                        exampleCounter = 0;
                        in_batches.emplace_back();
                        in_batches[batchIndex].reserve(inBatchSize);
                    }
                    sampleCounter = 0;
                }

                if (i >= audio.getNumSamplesPerChannel()-1 && sampleCounter != 0) {
                    for (uint32_t s=sampleCounter; s < net.InputSize(); ++s)
                        in_batches[batchIndex].emplace_back(0.f);
                    if (++exampleCounter >= examples_per_batch) {
                        batchIndex++;
                        exampleCounter = 0;
                        in_batches.emplace_back();
                        in_batches[batchIndex].reserve(inBatchSize);
                    }
                }
            }
        }

        if (in_batches.back().empty()) in_batches.pop_back();

    } else if (data_type == "TEXT_FILES") {

        bool fastSplit = StrToUpper(GLOBALS::config_map["FAST_SPLIT"]) == "TRUE";
        uint32_t splitSize = stoul(GLOBALS::config_map["SPLIT_SIZE"]);
        uint32_t minBatchSize = stoul(GLOBALS::config_map["MIN_BATCH_SIZE"]);
        std::vector<std::string> textFiles = ListAllFiles(data_file);
        std::vector<std::thread*> fileThreads(8, nullptr);
        std::array<std::atomic<bool>, 8> threadReady;
        std::fill(threadReady.begin(), threadReady.end(), true);
        std::atomic<uint32_t> batchIndex = 0;
        size_t batchesRes = textFiles.size() * 2;
        float halfExamplesPB = examples_per_batch * 0.5f;

        std::cout << "Counting batches ..." << std::endl;

        for (const auto& fileName : textFiles)
        {
            if (FileSize(fileName) > splitSize)
                batchesRes += CountStrInFile(fileName, GLOBALS::config_map["END_TXT_TAG"]);
        }

        if (batchesRes <= 0) HandleFatalError("Unable to find text files in "+data_file);

        token_batches.resize(batchesRes);

        Worderizer::LoadWordMap(word_map, GLOBALS::config_map["TOKENIZER"]);
        Worderizer::LoadSubChars(GLOBALS::DATA_FOLDER+SUBCHAR_FILE);

        auto tokenizeFunc = [&](uint32_t thread_index, uint32_t file_index, uint32_t batch_index)
        {
            std::vector<std::string> sequences;
            size_t fileSize = FileSize(textFiles[file_index]);
            uint32_t bIndex = batch_index;
            uint32_t sIndex = 0;

            if (fileSize > splitSize) {
                if (fastSplit) {
                    SplitStr(ReadFileStr(textFiles[file_index]), GLOBALS::config_map["END_TXT_TAG"], sequences);
                } else {
                    sequences.reserve(std::ceil(fileSize/halfExamplesPB));
                    SplitTextFile(textFiles[file_index], GLOBALS::config_map["END_TXT_TAG"], sequences);
                }
            } else {
                sequences.emplace_back(ReadFileStr(textFiles[file_index]));
            }

            for (const std::string& sequence: sequences)
            {
                if (sequence.length() < minBatchSize) continue;

                token_batches[bIndex].reserve(std::ceil(std::min(halfExamplesPB, sequence.length()*0.3f)));

                if (!Worderizer::StrToTokens(Worderizer::U8ToU32(sequence), token_batches[bIndex], word_map)) {
                    HandleFatalError("Failed to tokenize text in "+textFiles[file_index]);
                }

                if (token_batches[bIndex].size() > examples_per_batch) {

                    uint32_t extraBatches = std::ceil(token_batches[bIndex].size() / (float)examples_per_batch)-1;
                    uint32_t cIndex = examples_per_batch;

                    for (uint32_t i=0; i < extraBatches; ++i)
                    {
                        uint32_t numRemain = token_batches[bIndex].size() - cIndex;
                        uint32_t numToCopy = std::min(examples_per_batch, numRemain);

                        if (numRemain < minBatchSize) break;

                        auto startIndex = token_batches[bIndex].begin() + cIndex;
                        token_batches[batchIndex++].assign(startIndex, startIndex+numToCopy);

                        cIndex += numToCopy;
                    }

                    token_batches[bIndex].resize(examples_per_batch);
                }

                if (++sIndex < sequences.size()) bIndex = batchIndex++;
            }

            threadReady[thread_index] = true;
        };

        std::cout << "Tokenizing text ..." << std::endl;

        for (size_t i=0; i < textFiles.size(); ++i)
        {
            std::cout << "Reading text from " << textFiles[i] << std::endl;

            for (bool waitForThread=true; waitForThread; std::this_thread::sleep_for(10us))
            {
                for (size_t t=0; t < fileThreads.size(); ++t)
                {
                    if (threadReady[t]) {

                        if (fileThreads[t] != nullptr && fileThreads[t]->joinable()) {
                            fileThreads[t]->join();
                            delete fileThreads[t];
                        }

                        waitForThread = false;
                        threadReady[t] = false;
                        fileThreads[t] = new std::thread(tokenizeFunc,t,i,batchIndex++);
                        break;
                    }
                }
            }
        }

        for (size_t t=0; t < fileThreads.size(); ++t)
        {
            if (fileThreads[t] != nullptr) {
                if (fileThreads[t]->joinable()) fileThreads[t]->join();
                delete fileThreads[t];
            }
        }

        token_batches.resize(batchIndex);

    } else if (data_type == "BIN_OUT_IN" || data_type == "BIN_IN_OUT") {
        //TODO:
    } else if (data_type == "CSV_SEQUENCE") {
        //TODO:
    }

    if (in_batches.size() > 0)
        std::cout << "Loaded data into " << in_batches.size() << " batches" << std::endl;

    if (token_batches.size() > 0)
        std::cout << "Loaded text into " << token_batches.size() << " batches" << std::endl;
}
