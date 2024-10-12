#include "Engine.h"

using namespace std::chrono_literals;

Engine::Engine(const bool use_train_steps) :
    useTrainSteps(use_train_steps) {}

Engine::Engine(const std::vector<GLuint>& net_shape, std::string net_type, GLfloat min_weight, GLfloat max_weight,
               std::string in_act_func, std::string hid_act_func, std::string out_act_func, std::string mem_blend_func) :
    net(net_shape, net_type, min_weight, max_weight, in_act_func, hid_act_func, out_act_func, mem_blend_func) { Init(); }

Engine::Engine(std::string net_dir, std::string net_name) :
    net(net_dir, net_name) { Init(); }

void Engine::Init()
{
	assert(sizeof(Neuron) == 48);

	/*net.ConfigLearning(
        stof(GLOBALS::config_map["LEARN_RATE"]),
        stof(GLOBALS::config_map["MOMENTUM"]),
        stof(GLOBALS::config_map["TRAIN_STEPS"])
    );*/

	openGL.Initialize(net.LayerCount(), useTrainSteps);
}

void Engine::GetNetOutputs(std::vector<GLfloat>& dest)
{
    std::vector<Neuron> outLayer;
    outLayer.resize(net.OutputSize());
    dest.resize(net.OutputSize());

    openGL.CopyOutputLayer(net, outLayer);

    for (GLuint i=0; i < outLayer.size(); ++i)
        dest[i] = outLayer[i].actout;

    glFinish();
}

void Engine::TestNet()
{
    bool showIndexInfo = (
        GLOBALS::config_map["NET_TYPE"] == "WORD2VEC" ||
        GLOBALS::config_map["OUTPUT_FUNC"] == "INDEX_TO_ARRAY"
    );

    openGL.CopyNetToBuffers(net);
    openGL.AllocInputSSBO(net.InputSize(), NULL);
    openGL.AllocTargetSSBO(net.OutputSize(), NULL);
    openGL.AllocErrorSSBO(net.OutputSize());

    for (GLuint b=0; b < inBatches.size(); ++b)
    {
        std::cout << "Testing on batch " << b << " ..." << std::endl;

        GLfloat batchError = 0.0f;
        GLuint rightCount = 0;
        GLuint wrongCount = 0;
        GLuint exampleNum = inBatches[b].size() / net.InputSize();

        for (GLuint i=0; i < exampleNum; ++i)
        {
            openGL.CopyInputsToSSBO(net.InputSize(), &inBatches[b][net.InputSize()*i]);
            openGL.ForwardProp(net, GLOBALS::WORKGROUP_SIZE);

            if (showIndexInfo) {

                GLfloat* targets = &outBatches[b][net.OutputSize()*i];
                std::vector<GLfloat> outputs;
                GetNetOutputs(outputs);
                GLfloat highVal = 0.0f;
                GLuint outIndex = 0;
                GLuint targIndex = 0;

                for (GLuint o=0; o < outputs.size(); ++o)
                {
                    if (targets[o] == 1.0f) {
                        targIndex = o;
                        break;
                    }
                }

                for (GLuint o=0; o < outputs.size(); ++o)
                {
                    if (outputs[o] > highVal) {
                        highVal = outputs[o];
                        outIndex = o;
                    }
                }

                //std::cout << "Output Index: " << outIndex << ", Target Index: " << targIndex;
                if (outIndex == targIndex) {
                    rightCount++;
                    //std::cout << " (RIGHT)" << std::endl;
                } else {
                    wrongCount++;
                    //std::cout << " (WRONG)" << std::endl;
                }

            } else {

                std::vector<GLfloat> errors(net.OutputSize(), 0.0f);

                openGL.CopyTargetsToSSBO(net.OutputSize(), &outBatches[b][net.OutputSize()*i]);
                openGL.UpdateErrors(net, GLOBALS::WORKGROUP_SIZE);
                openGL.CopyErrorsFromSSBO(errors);
                openGL.ResetErrorSSBO();

                GLfloat avgOutErr = 0.0f;

                for (GLuint o=0; o < errors.size(); ++o) avgOutErr += errors[o];
                avgOutErr /= errors.size();
                batchError += avgOutErr;
                //std::cout << "Average output error: " << avgOutErr << std::endl;
            }
        }

        if (showIndexInfo) {
            float guesses = (float)rightCount + wrongCount;
            std::cout << "Right answers: " << rightCount;
            std::cout << " (" << (rightCount/guesses)*100.0f << "%)" << std::endl;
            std::cout << "Wrong answers: " << wrongCount;
            std::cout << " (" << (wrongCount/guesses)*100.0f << "%)" << std::endl;
        } else {
            batchError /= exampleNum;
            std::cout << "Average batch error: " << batchError << std::endl;
        }
    }
}

void Engine::ProcessData(const std::string& val, const std::string& func,
                         const GLfloat& div_arg, std::vector<GLfloat>& dest, GLuint array_size)
{
    if (func == "INDEX_TO_ARRAY") {

        GLuint setIndex = stoul(val);

        for (GLuint i=0; i < array_size; ++i)
        {
            if (i == setIndex) {
                dest.push_back(1.0f);
            } else {
                dest.push_back(0.0f);
            }
        }

    } else if (func == "DIV") {
        dest.push_back(stof(val)/div_arg);
    } else {
        dest.push_back(stof(val));
    }
}

void Engine::LoadDataBatches(std::string data_file, std::string data_type, std::string input_func,
                             std::string output_func, GLuint examples_per_batch)
{
    uint64_t inBatchSize = net.InputSize() * examples_per_batch;
    uint64_t outBatchSize = net.OutputSize() * examples_per_batch;
    uint64_t expectedIns = net.InputSize();
    uint64_t expectedOuts = net.OutputSize();
    std::string inFunc(input_func);
    std::string outFunc(output_func);
    float inFuncArg, outFuncArg;

    inBatches.clear();
    outBatches.clear();
    tokenBatches.clear();

    if (input_func == "INDEX_TO_ARRAY") {
        expectedIns = 1;
    } else if (input_func.rfind("DIV_", 0) == 0) {
        inFunc = "DIV";
        inFuncArg = stof(ExplodeStr(input_func, "_")[1]);
    }

    if (output_func == "INDEX_TO_ARRAY") {
        expectedOuts = 1;
    } else if (output_func.rfind("DIV_", 0) == 0) {
        outFunc = "DIV";
        outFuncArg = stof(ExplodeStr(output_func, "_")[1]);
    }

    if (data_type == "CSV_OUT_IN" || data_type == "CSV_IN_OUT")
    {
        std::ifstream ifs(data_file);
        std::string line;

        std::cout << "Loading data from file ..." << std::endl;

        if (ifs.is_open()) {

            std::getline(ifs, line);
            std::vector<std::string> vals(ExplodeStr(line, ","));
            uint64_t expectedVals = expectedIns + expectedOuts;
            uint64_t inputOffset = expectedOuts;
            uint64_t outputOffset = 0;
            uint64_t exampleCounter = 0;
            uint64_t batchIndex = 0;

            if (vals.size() != expectedVals) {
                std::stringstream errSS;
                errSS << "data file doesn't fit to net inputs/outputs" << std::endl;
                errSS << "expected values per line: " << std::to_string(expectedVals) << std::endl;
                errSS << "actual values per line: " << std::to_string(vals.size()) << std::endl;
                HandleFatalError(errSS.str());
            }

            if (data_type == "CSV_IN_OUT") {
                inputOffset = 0;
                outputOffset = expectedIns;
            }

            inBatches.emplace_back();
            outBatches.emplace_back();
            inBatches[0].reserve(inBatchSize);
            outBatches[0].reserve(outBatchSize);

            while (!ifs.eof())
            {
                std::getline(ifs, line);
                vals = ExplodeStr(line, ",");
                exampleCounter++;

                if (line == "") break;

                if (vals.size() != expectedVals)
                    HandleFatalError("csv file is corrupted!");

                for (GLuint i=0; i < expectedIns; ++i)
                    ProcessData(vals[inputOffset+i], inFunc, inFuncArg, inBatches[batchIndex], net.InputSize());

                for (GLuint i=0; i < expectedOuts; ++i)
                    ProcessData(vals[outputOffset+i], outFunc, outFuncArg, outBatches[batchIndex], net.OutputSize());

                if (exampleCounter >= examples_per_batch) {
                    batchIndex++;
                    exampleCounter = 0;
                    inBatches.emplace_back();
                    outBatches.emplace_back();
                    inBatches[batchIndex].reserve(inBatchSize);
                    outBatches[batchIndex].reserve(outBatchSize);
                }
            }

            if (inBatches.back().empty()) {
                inBatches.pop_back();
                outBatches.pop_back();
            }

            ifs.close();
        } else {
            HandleFatalError("failed to open file: "+data_file);
        }

        std::cout << "Loaded data into " << inBatches.size() << " batches" << std::endl;

    } else if (data_type == "TEXT_FILES") {

        bool fastSplit = StrToUpper(GLOBALS::config_map["FAST_SPLIT"]) == "TRUE";
        uint32_t splitSize = stoul(GLOBALS::config_map["SPLIT_SIZE"]);
        uint32_t minBatchSize = stoul(GLOBALS::config_map["MIN_BATCH_SIZE"]);
        std::vector<std::string> textFiles = ListFiles(data_file);
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

        tokenBatches.resize(batchesRes);

        Worderizer::LoadWordMap(wordMap, GLOBALS::config_map["TOKENIZER"]);

        auto tokenizeFunc = [&](uint32_t thread_index, uint32_t file_index, uint32_t batch_index)
        {
            std::vector<std::string> sequences;
            size_t fileSize = FileSize(textFiles[file_index]);
            uint32_t bIndex = batch_index;
            uint32_t sIndex = 0;

            if (fileSize > splitSize) {
                if (fastSplit) {
                    SplitText(ReadFileStr(textFiles[file_index]), GLOBALS::config_map["END_TXT_TAG"], sequences);
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

                tokenBatches[bIndex].reserve(std::ceil(std::min(halfExamplesPB, sequence.length()*0.3f)));

                if (!Worderizer::StrToTokens(Worderizer::U8ToU32(sequence), tokenBatches[bIndex], wordMap)) {
                    HandleFatalError("Failed to tokenize text in "+textFiles[file_index]);
                }

                if (tokenBatches[bIndex].size() > examples_per_batch) {

                    uint32_t extraBatches = std::ceil(tokenBatches[bIndex].size() / (float)examples_per_batch)-1;
                    uint32_t cIndex = examples_per_batch;

                    for (uint32_t i=0; i < extraBatches; ++i)
                    {
                        uint32_t numRemain = tokenBatches[bIndex].size() - cIndex;
                        uint32_t numToCopy = std::min(examples_per_batch, numRemain);

                        if (numRemain < minBatchSize) break;

                        auto startIndex = tokenBatches[bIndex].begin() + cIndex;
                        tokenBatches[batchIndex++].assign(startIndex, startIndex+numToCopy);

                        cIndex += numToCopy;
                    }

                    tokenBatches[bIndex].resize(examples_per_batch);
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

        tokenBatches.resize(batchIndex);

        std::cout << "Loaded text into " << tokenBatches.size() << " batches" << std::endl;

    } else if (data_type == "BIN_OUT_IN" || data_type == "BIN_IN_OUT") {
        //TODO:
    } else if (data_type == "CSV_SEQUENCE") {
        //TODO:
    }
}

void Engine::CheckTextFiles(const std::string& data_dir, float min_byte_token_ratio)
{
    std::vector<uint32_t> tokens;
    std::vector<std::string> textFiles = ListFiles(data_dir);

    if (wordMap.empty())
        Worderizer::LoadWordMap(wordMap, GLOBALS::config_map["TOKENIZER"]);

    for (size_t i=0; i < textFiles.size(); ++i)
    {
        std::string textStr = ReadFileStr(textFiles[i]);
        tokens.clear();

        if (Worderizer::StrToTokens(Worderizer::U8ToU32(textStr), tokens, wordMap)) {
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

void Engine::TrainNet(const GLuint train_epochs, const GLuint train_steps)
{
    if (inBatches.size() != outBatches.size())
        HandleFatalError("size of input and output batches must be the same");

    auto rd = std::random_device{};
    auto rng = std::default_random_engine{ rd() };

    GLuint trainStep = 0;

    std::vector<GLfloat> errors(net.OutputSize(), 0.0f);
    std::vector<GLuint> batchIndices(inBatches.size());
    std::iota(batchIndices.begin(), batchIndices.end(), 0);

    openGL.CopyNetToBuffers(net);
    openGL.AllocInputSSBO(net.InputSize(), NULL);
    openGL.AllocTargetSSBO(net.OutputSize(), NULL);
    openGL.AllocErrorSSBO(net.OutputSize());

    for (GLuint e=0; e < train_epochs; ++e)
    {
        std::cout << std::endl << "Epoch " << (e+1) << std::endl;
        GLfloat epochError = 0.0f;

        std::shuffle(batchIndices.begin(), batchIndices.end(), rng);

        for (GLuint b=0; b < inBatches.size(); ++b)
        {
            GLuint exampleNum = inBatches[batchIndices[b]].size() / net.InputSize();
            GLfloat batchError = 0.0f;

            std::cout << "Training on batch " << (b+1) << " with " << exampleNum << " examples" << std::endl;

            for (GLuint i=0; i < exampleNum; ++i)
            {
                openGL.CopyInputsToSSBO(net.InputSize(), &inBatches[batchIndices[b]][net.InputSize()*i]);
                openGL.ForwardProp(net, GLOBALS::WORKGROUP_SIZE);
                openGL.CopyTargetsToSSBO(net.OutputSize(), &outBatches[batchIndices[b]][net.OutputSize()*i]);
                openGL.BackwardProp(net, GLOBALS::WORKGROUP_SIZE);

                if (useTrainSteps) {
                    if (++trainStep >= train_steps) {
                        openGL.UpdateWeights(net, GLOBALS::WORKGROUP_SIZE);
                        openGL.UpdateNeurons(net, GLOBALS::WORKGROUP_SIZE);
                        trainStep = 0;
                    }
                } else {
                    openGL.UpdateWeights(net, GLOBALS::WORKGROUP_SIZE);
                }
            }

            openGL.CopyErrorsFromSSBO(errors);

            for (GLuint i=0; i < errors.size(); ++i)
                batchError += errors[i] / exampleNum;

            batchError /= errors.size();
            epochError += batchError;
            std::cout << "Average batch error: " << batchError << std::endl;

            openGL.ResetErrorSSBO();
            openGL.ResetNetMem(net, GLOBALS::WORKGROUP_SIZE);
        }

        epochError /= inBatches.size();
        std::cout << "Average epoch error: " << epochError << std::endl;
    }

    openGL.CopyNetFromBuffers(net);
}
void Engine::TrainTextNet(const GLuint train_epochs, const GLuint train_steps)
{
    if (tokenBatches.empty()) {
        HandleFatalError("tokenized training data wasn't generated");
    } else if (GLOBALS::config_map["LOSS_FUNC"] != "CROSS_ENTROPY") {
        HandleFatalError("text net must use cross-entropy loss function");
    }

    auto rd = std::random_device{};
    auto rng = std::default_random_engine{ rd() };

    uint32_t trainStep = 0;
    uint32_t inputIndex = 0;
    uint32_t lastInIndex = 0;
    uint32_t targetIndex = 0;
    uint32_t lastOutIndex = 0;

    std::vector<GLfloat> inputs(net.InputSize(), 0.0f);
    std::vector<GLfloat> targets(net.OutputSize(), 0.0f);
    std::vector<GLfloat> errors(net.OutputSize(), 0.0f);
    std::vector<GLuint> batchIndices(tokenBatches.size());
    std::iota(batchIndices.begin(), batchIndices.end(), 0);

    openGL.CopyNetToBuffers(net);
    openGL.AllocInputSSBO(net.InputSize(), inputs.data());
    openGL.AllocTargetSSBO(net.OutputSize(), targets.data());
    openGL.AllocErrorSSBO(net.OutputSize());

    for (GLuint e=0; e < train_epochs; ++e)
    {
        std::cout << std::endl << "Epoch " << (e+1) << std::endl;

        std::shuffle(batchIndices.begin(), batchIndices.end(), rng);

        uint32_t batchCount = tokenBatches.size();
        GLfloat epochError = 0.0f;

        for (GLuint b=0; b < tokenBatches.size(); ++b)
        {
            std::vector<uint32_t>& tokens(tokenBatches[batchIndices[b]]);
            GLuint exampleMax = tokens.size() - 1;
            GLuint percDoneInt = 0;
            GLfloat batchError = 0.0f;

            if (tokens.size() < 3) {
                std::cout << "Skipping batch " << (b+1) << " (too small) " << tokens.size() << std::endl;
                batchCount--;
                continue;
            }

            std::cout << "Training on batch " << (b+1) << " with " << tokens.size() << " tokens" << std::endl;

            for (GLuint i=0; i < exampleMax; ++i)
            {
                inputIndex = tokens[i];
                targetIndex = tokens[i+1];

                openGL.CopyIndexInToSSBO(inputIndex, lastInIndex, inputs);
                openGL.ForwardProp(net, GLOBALS::WORKGROUP_SIZE);
                openGL.CopyIndexOutToSSBO(targetIndex, lastOutIndex, targets);
                openGL.BackwardPropFast(net, errors, targetIndex, GLOBALS::WORKGROUP_SIZE);

                lastInIndex = inputIndex;
                lastOutIndex = targetIndex;

                if (useTrainSteps) {
                    if (++trainStep >= train_steps) {
                        openGL.UpdateWeights(net, GLOBALS::WORKGROUP_SIZE);
                        openGL.UpdateNeurons(net, GLOBALS::WORKGROUP_SIZE);
                        trainStep = 0;
                    }
                } else {
                    openGL.UpdateWeights(net, GLOBALS::WORKGROUP_SIZE);
                }

                percDoneInt = ((float)i / exampleMax) * 100.0f;

                if (percDoneInt < 10) {
                    std::cout << "Progress: " << percDoneInt << "% \r" << std::flush;
                } else if (percDoneInt < 100) {
                    std::cout << "Progress: " << percDoneInt << "%\r" << std::flush;
                } else {
                    std::cout << "Progress: 100%" << std::endl;
                }
            }

            for (GLuint i=0; i < errors.size(); ++i)
                batchError += errors[i] / exampleMax;

            batchError /= errors.size();
            epochError += batchError;
            std::cout << "Average batch error: " << batchError << std::endl;

            std::fill(errors.begin(), errors.end(), 0.0f);
            openGL.ResetNetMem(net, GLOBALS::WORKGROUP_SIZE);
        }

        epochError /= batchCount;
        std::cout << "Average epoch error: " << epochError << std::endl;
    }

    openGL.CopyNetFromBuffers(net);
}
