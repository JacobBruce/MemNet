#include "Engine.h"

Engine::Engine(const bool use_train_steps) :
    useTrainSteps(use_train_steps)
{
	assert(sizeof(Neuron) == 48);

    if (GLOBALS::config_map["LOSS_FUNC"] == "BIN_FOCAL_LOSS") {
        GLOBALS::POS_LABEL = 1.f;
        GLOBALS::NEG_LABEL = -1.f;
    } else {
        GLOBALS::POS_LABEL = 1.f;
        GLOBALS::NEG_LABEL = 0.f;
    }

	if (GLOBALS::config_map.find("POS_WEIGHT") == GLOBALS::config_map.end())
        GLOBALS::config_map["POS_WEIGHT"] = "1.0";

	if (GLOBALS::config_map.find("NEG_WEIGHT") == GLOBALS::config_map.end())
        GLOBALS::config_map["NEG_WEIGHT"] = "1.0";

	if (GLOBALS::config_map.find("FL_ALPHA") == GLOBALS::config_map.end())
        GLOBALS::config_map["FL_ALPHA"] = "0.25";

	if (GLOBALS::config_map.find("FL_GAMMA") == GLOBALS::config_map.end())
        GLOBALS::config_map["FL_GAMMA"] = "2.0";
}

void Engine::SetNet(uint32_t net_index)
{
    activeNetIndex = net_index;
}

MemNet& Engine::GetNet()
{
    return nets[activeNetIndex];
}

GL& Engine::GetGL()
{
    return glVec[activeNetIndex];
}

void Engine::InitNet(MemNet& net)
{
	glVec.emplace_back();

	glVec.back().Initialize(net.ActivationFuncs(), useTrainSteps);

    glVec.back().CopyNetToBuffers(net);

	if (GLOBALS::config_map.find("BP_DEPTH") == GLOBALS::config_map.end()) {
        GLOBALS::BP_DEPTH = net.LayerCount();
	} else if (StrToUpper(GLOBALS::config_map["BP_DEPTH"]) == "ALL") {
	    GLOBALS::BP_DEPTH = net.LayerCount();
	} else {
        GLOBALS::BP_DEPTH = stoul(GLOBALS::config_map["BP_DEPTH"]);

        if (GLOBALS::BP_DEPTH == 0)
            HandleFatalError("BP_DEPTH cannot be 0");
        if (GLOBALS::BP_DEPTH > net.LayerCount())
            HandleFatalError("BP_DEPTH is greater than net layer count");
	}

    if (GLOBALS::BP_DEPTH == net.LayerCount()) {
        GLOBALS::BP_STOP = 0;
    } else {
        GLOBALS::BP_STOP = net.LayerCount() - (GLOBALS::BP_DEPTH + 1);
    }
}

void Engine::LoadNet(std::string net_dir, std::string net_name)
{
    SetNet(nets.size());
    nets.emplace_back(net_dir, net_name);
    InitNet(nets.back());
}

void Engine::GenNet(const std::vector<uint32_t>& net_shape, std::string net_type, float4 init_data,
            std::vector<std::string>& act_funcs, std::vector<std::string>& mem_layers, std::string mem_blend_func)
{
    SetNet(nets.size());
    nets.emplace_back(net_shape, net_type, init_data, act_funcs, mem_layers, mem_blend_func);
    InitNet(nets.back());
}

void Engine::CopyLayerOutputs(MemNet& net, std::vector<GLfloat>& dest, GLuint layer_index)
{
    GLuint layerSize = net.LayerSize(layer_index);
    Neuron* neurons(net.LayerNeurons(layer_index));
    dest.resize(layerSize);

    GetGL().CopyNeuronsFromSSBO(net, layer_index);

    for (GLuint i=0; i < layerSize; ++i)
        dest[i] = neurons[i].actout;
}

void Engine::GetNetOutputs(MemNet& net, std::vector<GLfloat>& dest)
{
    CopyLayerOutputs(net, dest, net.LayerCount()-1);
}

void Engine::LoadDataset(std::string data_file, std::string data_type, std::string input_func,
                         std::string output_func, uint32_t examples_per_batch)
{
    if (GLOBALS::config_map["DATA_TYPE"] == "CSV_FILES") {

        std::cout << "Loading data from " << data_file << std::endl;

        std::vector<std::string> fileLines(ReadFileLines(data_file));

        for (size_t i=0; i+1 < fileLines.size(); i+=2)
        {
            std::string& inputFile(fileLines[i]);
            std::string& outputFile(fileLines[i+1]);

            LoadDataFiles(inputFile, outputFile, input_func, output_func,
                          examples_per_batch, GetNet(), inBatches, outBatches);
        }

        if (inBatches.back().empty()) inBatches.pop_back();
        if (outBatches.back().empty()) outBatches.pop_back();

        if (inBatches.size() > 0)
            std::cout << "Loaded data into " << inBatches.size() << " batches" << std::endl;

    } else {
        LoadDataBatches(data_file, data_type, input_func, output_func, examples_per_batch,
                        GetNet(), inBatches, outBatches, tokenBatches, wordMap);
    }
}

void Engine::SaveNet(std::string net_dir, std::string net_name)
{
    GetGL().CopyNetFromBuffers(GetNet());

    GetNet().Save(net_dir, net_name);
}

void Engine::RunNet()
{
    GL& openGL(GetGL());
    MemNet& net(GetNet());

    std::cout << "Running model ..." << std::endl;

    openGL.AllocInputSSBO(net.InputSize(), NULL);

    bool runOnce = StrIsTrue(GLOBALS::config_map["RUN_ONCE"]);
    bool appendOut = StrEndsWith(GLOBALS::config_map["OUT_TARGET"], "_APPEND");
    uint8_t outTarget = GLOBALS::config_map["OUT_TARGET"] == "CONSOLE" ? 0 : 1;
    std::filesystem::file_time_type lwt;

    std::string outputStr;
    std::vector<GLfloat> outputs;

    while (true)
    {
        if (FileExists(GLOBALS::config_map["IN_FILE"])) {
            auto wt = std::filesystem::last_write_time(GLOBALS::config_map["IN_FILE"]);
            if (lwt == wt) {
                std::this_thread::sleep_for(1ms);
                continue;
            }
            lwt = wt;
        } else {
            std::cout << "Input file not found, exiting." << std::endl;
            break;
        }

        LoadDataset(
            GLOBALS::config_map["IN_FILE"], "CSV_IN",
            GLOBALS::config_map["INPUT_FUNC"], "NONE", 1
        );

        outputStr.clear();

        for (size_t i=0; i < inBatches.size(); ++i)
        {
            openGL.CopyInputsToSSBO(net.InputSize(), &inBatches[i][0]);
            openGL.ForwardProp(net, GLOBALS::WORKGROUP_SIZE);

            GetNetOutputs(net, outputs);

            for (GLuint n=0; n < outputs.size(); ++n)
                outputStr += std::to_string(outputs[n]) + ",";

            outputStr = TrimR(outputStr, ',') + "\n";
        }

        if (outTarget == 0) {
            std::cout << std::endl << "OUTPUT:\n" << outputStr << std::endl;
        } else {
            WriteFileStr(GLOBALS::config_map["OUT_FILE"], outputStr, appendOut);
            std::cout << "Saved net outputs to file." << std::endl;
        }

        if (runOnce) break;
    }
}

void Engine::TestNet()
{
    GL& openGL(GetGL());
    MemNet& net(GetNet());

    bool showIndexInfo = StrStartsWith(GLOBALS::config_map["OUTPUT_FUNC"], "INDEX_TO_");
    bool isW2Vnet = net.NetType() == "WORD_2_VEC";
    bool isTextNet = GLOBALS::config_map["INPUT_FUNC"] == "TOKENIZE";
    bool isTextAutoEnc = net.NetType() == "AUTO_ENCODER" && isTextNet;
    uint32_t inOutMax = std::max(net.InputSize(), net.OutputSize());

    std::vector<GLfloat> outputs;
    std::vector<GLfloat> zeros(inOutMax, 0.0f);
    std::vector<GLfloat> ones(inOutMax, 1.0f);
    std::vector<GLuint> batchIndices(tokenBatches.size());
    std::iota(batchIndices.begin(), batchIndices.end(), 0);

    openGL.AllocInputSSBO(net.InputSize(), zeros.data());
    openGL.AllocTargetSSBO(net.OutputSize(), zeros.data());
    openGL.AllocErrorSSBO(net.OutputSize(), zeros.data());

    if (isW2Vnet) {

        for (GLuint b=0; b < tokenBatches.size(); ++b)
        {
            std::vector<uint32_t>& tokens(tokenBatches[batchIndices[b]]);
            GLuint exampleMax = tokens.size() - 1;
            GLuint exampleNum = exampleMax - 1;
            GLuint rightCountF = 0;
            GLuint rightCountP = 0;
            GLuint wrongCountF = 0;
            GLuint wrongCountP = 0;
            GLuint lastInIndex = 0;

            std::cout << "Testing on batch " << (b+1) << " with " << exampleNum << " examples ..." << std::endl;

            for (GLuint i=1; i < exampleNum; ++i)
            {
                GLuint inputIndex = tokens[i];
                GLuint targIndex1 = tokens[i+1];
                GLuint targIndex2 = net.InputSize() + tokens[i-1];

                openGL.CopyIndexInToSSBO(inputIndex, lastInIndex, ones);
                openGL.ForwardProp(net, GLOBALS::WORKGROUP_SIZE);

                GetNetOutputs(net, outputs);

                lastInIndex = inputIndex;
                GLuint outIndex = 0;
                GLfloat highVal = 0.0f;

                for (GLuint o=0; o < net.InputSize(); ++o)
                {
                    if (outputs[o] > highVal) {
                        highVal = outputs[o];
                        outIndex = o;
                    }
                }

                if (outIndex == targIndex1) {
                    rightCountF++;
                } else {
                    wrongCountF++;
                }

                outIndex = 0;
                highVal = 0.0f;

                for (GLuint o=net.InputSize(); o < net.OutputSize(); ++o)
                {
                    if (outputs[o] > highVal) {
                        highVal = outputs[o];
                        outIndex = o;
                    }
                }

                if (outIndex == targIndex2) {
                    rightCountP++;
                } else {
                    wrongCountP++;
                }
            }

            if (net.MemBlendFunc() != "0")
                openGL.ResetNetMem(net, GLOBALS::WORKGROUP_SIZE);

            float guesses = (float)rightCountF + wrongCountF;
            std::cout << "Right future predictions: " << rightCountF;
            std::cout << " (" << (rightCountF/guesses)*100.0f << "%)" << std::endl;
            std::cout << "Wrong future predictions: " << wrongCountF;
            std::cout << " (" << (wrongCountF/guesses)*100.0f << "%)" << std::endl;

            guesses = (float)rightCountP + wrongCountP;
            std::cout << "Right past predictions: " << rightCountP;
            std::cout << " (" << (rightCountP/guesses)*100.0f << "%)" << std::endl;
            std::cout << "Wrong past predictions: " << wrongCountP;
            std::cout << " (" << (wrongCountP/guesses)*100.0f << "%)" << std::endl;
        }

    } else if (isTextAutoEnc) {

        for (GLuint b=0; b < tokenBatches.size(); ++b)
        {
            std::vector<uint32_t>& tokens(tokenBatches[batchIndices[b]]);
            GLuint exampleMax = tokens.size() - 1;
            GLuint exampleNum = exampleMax - 1;
            GLuint rightCount = 0;
            GLuint wrongCount = 0;
            GLuint lastInIndex = 0;

            std::cout << "Testing on batch " << (b+1) << " with " << exampleNum << " examples ..." << std::endl;

            for (GLuint i=1; i < exampleNum; ++i)
            {
                openGL.CopyIndexInToSSBO(tokens[i], lastInIndex, ones);
                openGL.ForwardProp(net, GLOBALS::WORKGROUP_SIZE);

                GetNetOutputs(net, outputs);

                lastInIndex = tokens[i];
                GLuint outIndex = 0;
                GLfloat highVal = 0.0f;

                for (GLuint o=0; o < net.InputSize(); ++o)
                {
                    if (outputs[o] > highVal) {
                        highVal = outputs[o];
                        outIndex = o;
                    }
                }

                if (outIndex == lastInIndex) {
                    rightCount++;
                } else {
                    wrongCount++;
                }
            }

            if (net.MemBlendFunc() != "0")
                openGL.ResetNetMem(net, GLOBALS::WORKGROUP_SIZE);

            float guesses = (float)rightCount + wrongCount;
            std::cout << "Right answers: " << rightCount;
            std::cout << " (" << (rightCount/guesses)*100.0f << "%)" << std::endl;
            std::cout << "Wrong answers: " << wrongCount;
            std::cout << " (" << (wrongCount/guesses)*100.0f << "%)" << std::endl;
        }

    } else {

        for (GLuint b=0; b < inBatches.size(); ++b)
        {
            GLfloat batchError = 0.0f;
            GLuint rightCount = 0;
            GLuint wrongCount = 0;
            GLuint exampleNum = inBatches[b].size() / net.InputSize();

            std::cout << "Testing on batch " << (b+1) << " with " << exampleNum << " examples ..." << std::endl;

            for (GLuint i=0; i < exampleNum; ++i)
            {
                openGL.CopyInputsToSSBO(net.InputSize(), &inBatches[b][net.InputSize()*i]);
                openGL.ForwardProp(net, GLOBALS::WORKGROUP_SIZE);

                if (showIndexInfo) {

                    GLfloat* targets = &outBatches[b][net.OutputSize()*i];
                    GetNetOutputs(net, outputs);
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

                    if (outIndex == targIndex) {
                        rightCount++;
                    } else {
                        wrongCount++;
                    }

                } else {

                    std::vector<GLfloat> errors(net.OutputSize(), 0.0f);

                    if (net.NetType() == "AUTO_ENCODER" || GLOBALS::config_map["DATA_TYPE"] == "CSV_IN") {
                        openGL.CopyTargetsToSSBO(net.OutputSize(), &inBatches[b][net.OutputSize()*i]);
                    } else if (net.NetType() == "AUTO_ENC_MEM") {
                        openGL.CopyInOutsToTargs(net.InputSize(), GLOBALS::WORKGROUP_SIZE);
                    } else {
                        openGL.CopyTargetsToSSBO(net.OutputSize(), &outBatches[b][net.OutputSize()*i]);
                    }

                    openGL.UpdateErrors(net, GLOBALS::WORKGROUP_SIZE);
                    openGL.CopyErrorsFromSSBO(errors);
                    openGL.ResetErrorSSBO();

                    GLfloat avgOutErr = 0.0f;

                    for (GLuint o=0; o < errors.size(); ++o) avgOutErr += errors[o];
                    avgOutErr /= errors.size();
                    batchError += avgOutErr;
                }
            }

            if (net.MemBlendFunc() != "0")
                openGL.ResetNetMem(net, GLOBALS::WORKGROUP_SIZE);

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
}

void Engine::TrainNet(const GLuint train_epochs, const GLuint train_steps)
{
    GL& openGL(GetGL());
    MemNet& net(GetNet());

    if (inBatches.empty()) {
        HandleFatalError("no input data was loaded!");
    } else if (inBatches.size() != outBatches.size()) {
        HandleFatalError("size of input and output batches must be the same");
    }

    auto rd = std::random_device{};
    auto rng = std::default_random_engine{ rd() };

    uint32_t trainStep = 0;
    uint32_t inOutMax = std::max(net.InputSize(), net.OutputSize());

    std::vector<GLfloat> zeros(inOutMax, 0.0f);
    std::vector<GLfloat> errors(net.OutputSize(), 0.0f);
    std::vector<GLuint> batchIndices(inBatches.size());
    std::iota(batchIndices.begin(), batchIndices.end(), 0);

    openGL.AllocInputSSBO(net.InputSize(), zeros.data());
    openGL.AllocTargetSSBO(net.OutputSize(), zeros.data());
    openGL.AllocErrorSSBO(net.OutputSize(), zeros.data());

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

            if (net.MemBlendFunc() != "0")
                openGL.ResetNetMem(net, GLOBALS::WORKGROUP_SIZE);
        }

        epochError /= inBatches.size();
        std::cout << "Average epoch error: " << epochError << std::endl;
    }
}

void Engine::TrainTextAutoEncoder(const GLuint train_epochs, const GLuint train_steps)
{
    GL& openGL(GetGL());
    MemNet& net(GetNet());

    //TODO: support multi-token input with multiple output targets
    if (tokenBatches.empty()) {
        HandleFatalError("tokenized training data wasn't generated");
    } else if (net.InputSize() != wordMap.size()) {
        HandleFatalError("number of tokenizer words doesn't match net input count");
    //} else if (!StrEndsWith(GLOBALS::config_map["LOSS_FUNC"], "CROSS_ENTROPY")) {
    //    HandleFatalError("text net must use cross-entropy loss function");
    } else if (net.InputSize() != net.OutputSize()) {
        HandleFatalError("size of input layer must be the same as output layer");
    }

    auto rd = std::random_device{};
    auto rng = std::default_random_engine{ rd() };

    uint32_t trainStep = 0;
    uint32_t lastInIndex = 0;

    bool encRawInputs = net.NetType() != "AUTO_ENC_MEM";

    std::vector<GLfloat> targets(net.OutputSize(), GLOBALS::NEG_LABEL);
    std::vector<GLfloat> zeros(net.InputSize(), 0.0f);
    std::vector<GLfloat> ones(net.InputSize(), 1.0f);
    std::vector<GLfloat> errors(net.OutputSize(), 0.0f);
    std::vector<GLuint> batchIndices(tokenBatches.size());
    std::iota(batchIndices.begin(), batchIndices.end(), 0);

    openGL.AllocInputSSBO(net.InputSize(), zeros.data());
    openGL.AllocTargetSSBO(net.OutputSize(), targets.data());
    openGL.AllocErrorSSBO(net.OutputSize(), zeros.data());

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
                openGL.CopyIndexInToSSBO(tokens[i], lastInIndex, ones);
                openGL.ForwardProp(net, GLOBALS::WORKGROUP_SIZE);

                if (encRawInputs) {
                    openGL.CopyIndexOutToSSBO(tokens[i], lastInIndex);
                } else {
                    openGL.CopyInOutsToTargs(net.InputSize(), GLOBALS::WORKGROUP_SIZE);
                }

                openGL.BackwardProp(net, GLOBALS::WORKGROUP_SIZE);
                //openGL.BackwardPropFast(net, errors, inputIndexes, GLOBALS::WORKGROUP_SIZE);

                lastInIndex = tokens[i];

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

            openGL.CopyErrorsFromSSBO(errors);

            for (GLuint i=0; i < errors.size(); ++i)
                batchError += errors[i] / exampleMax;

            batchError /= errors.size();
            epochError += batchError;
            std::cout << "Average batch error: " << batchError << std::endl;

            openGL.ResetErrorSSBO();

            if (net.MemBlendFunc() != "0")
                openGL.ResetNetMem(net, GLOBALS::WORKGROUP_SIZE);
        }

        epochError /= batchCount;
        std::cout << "Average epoch error: " << epochError << std::endl;
    }
}

void Engine::TrainAutoEncoder(const GLuint train_epochs, const GLuint train_steps)
{
    GL& openGL(GetGL());
    MemNet& net(GetNet());

    if (net.InputSize() != net.OutputSize()) {
        HandleFatalError("size of input layer must be the same as output layer");
    } else if (inBatches.empty()) {
        HandleFatalError("no input data was loaded!");
    }

    auto rd = std::random_device{};
    auto rng = std::default_random_engine{ rd() };

    uint32_t trainStep = 0;

    bool encRawInputs = net.NetType() != "AUTO_ENC_MEM";

    std::vector<GLfloat> zeros(net.OutputSize(), 0.0f);
    std::vector<GLfloat> errors(net.OutputSize(), 0.0f);
    std::vector<GLuint> batchIndices(inBatches.size());
    std::iota(batchIndices.begin(), batchIndices.end(), 0);

    openGL.AllocInputSSBO(net.InputSize(), zeros.data());
    openGL.AllocTargetSSBO(net.OutputSize(), zeros.data());
    openGL.AllocErrorSSBO(net.OutputSize(), zeros.data());

    for (GLuint e=0; e < train_epochs; ++e)
    {
        std::cout << std::endl << "Epoch " << (e+1) << std::endl;
        GLfloat epochError = 0.0f;

        std::shuffle(batchIndices.begin(), batchIndices.end(), rng);

        for (GLuint b=0; b < inBatches.size(); ++b)
        {
            std::vector<GLfloat>& batch(inBatches[batchIndices[b]]);
            if (batch.size() % net.InputSize() != 0) {
                std::cout << "batch.size(): " << batch.size() << std::endl;
                HandleFatalError("batch error");
            }
            GLuint exampleNum = batch.size() / net.InputSize();
            GLfloat batchError = 0.0f;

            std::cout << "Training on batch " << (b+1) << " with " << exampleNum << " examples" << std::endl;

            for (GLuint i=0; i < exampleNum; ++i)
            {
                openGL.CopyInputsToSSBO(net.InputSize(), &batch[net.InputSize()*i]);
                openGL.ForwardProp(net, GLOBALS::WORKGROUP_SIZE);

                if (encRawInputs) {
                    openGL.CopyTargetsToSSBO(net.OutputSize(), &batch[net.OutputSize()*i]);
                } else {
                    openGL.CopyInOutsToTargs(net.InputSize(), GLOBALS::WORKGROUP_SIZE);
                }

                openGL.BackwardProp(net, GLOBALS::WORKGROUP_SIZE);

                if (useTrainSteps) {
                    if (++trainStep >= train_steps) {
                        openGL.UpdateWeights(net, GLOBALS::WORKGROUP_SIZE);
                        openGL.UpdateNeurons(net, GLOBALS::WORKGROUP_SIZE);
                        trainStep = 0;
                    }
                } else {
                    openGL.UpdateWeights(net, GLOBALS::WORKGROUP_SIZE);
                    //openGL.ResetNeurons(net, GLOBALS::WORKGROUP_SIZE);
                }
            }

            openGL.CopyErrorsFromSSBO(errors);

            for (GLuint i=0; i < errors.size(); ++i)
                batchError += errors[i] / exampleNum;

            batchError /= errors.size();
            epochError += batchError;
            std::cout << "Average batch error: " << batchError << std::endl;

            openGL.ResetErrorSSBO();

            if (net.MemBlendFunc() != "0")
                openGL.ResetNetMem(net, GLOBALS::WORKGROUP_SIZE);
        }

        epochError /= inBatches.size();
        std::cout << "Average epoch error: " << epochError << std::endl;
    }
}

void Engine::TrainTextNet(const GLuint train_epochs, const GLuint train_steps)
{
    GL& openGL(GetGL());
    MemNet& net(GetNet());

    if (tokenBatches.empty()) {
        HandleFatalError("tokenized training data wasn't generated");
    } else if (net.InputSize() != wordMap.size()) {
        HandleFatalError("number of tokenizer words doesn't match net input count");
    } else if (!StrEndsWith(GLOBALS::config_map["LOSS_FUNC"], "CROSS_ENTROPY")) {
        HandleFatalError("text net must use cross-entropy loss function");
    }

    auto rd = std::random_device{};
    auto rng = std::default_random_engine{ rd() };

    uint32_t trainStep = 0;
    uint32_t inputIndex = 0;
    uint32_t lastInIndex = 0;
    uint32_t targetIndex = 0;
    uint32_t lastOutIndex = 0;
    uint32_t inOutMax = std::max(net.InputSize(), net.OutputSize());

    std::vector<GLfloat> targets(net.OutputSize(), GLOBALS::NEG_LABEL);
    std::vector<GLfloat> zeros(inOutMax, 0.0f);
    std::vector<GLfloat> ones(inOutMax, 1.0f);
    std::vector<GLfloat> errors(net.OutputSize(), 0.0f);
    std::vector<GLuint> batchIndices(tokenBatches.size());
    std::iota(batchIndices.begin(), batchIndices.end(), 0);

    openGL.AllocInputSSBO(net.InputSize(), zeros.data());
    openGL.AllocTargetSSBO(net.OutputSize(), targets.data());
    openGL.AllocErrorSSBO(net.OutputSize(), zeros.data());

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

                openGL.CopyIndexInToSSBO(inputIndex, lastInIndex, ones);
                openGL.ForwardProp(net, GLOBALS::WORKGROUP_SIZE);
                openGL.CopyIndexOutToSSBO(targetIndex, lastOutIndex);
                openGL.BackwardProp(net, GLOBALS::WORKGROUP_SIZE);
                //openGL.BackwardPropFast(net, errors, targetIndexes, GLOBALS::WORKGROUP_SIZE);

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

            openGL.CopyErrorsFromSSBO(errors);

            for (GLuint i=0; i < errors.size(); ++i)
                batchError += errors[i] / exampleMax;

            batchError /= errors.size();
            epochError += batchError;
            std::cout << "Average batch error: " << batchError << std::endl;

            openGL.ResetErrorSSBO();

            if (net.MemBlendFunc() != "0")
                openGL.ResetNetMem(net, GLOBALS::WORKGROUP_SIZE);
        }

        epochError /= batchCount;
        std::cout << "Average epoch error: " << epochError << std::endl;
    }
}

void Engine::TrainWord2Vec(const GLuint train_epochs, const GLuint train_steps)
{
    GL& openGL(GetGL());
    MemNet& net(GetNet());

    if (tokenBatches.empty()) {
        HandleFatalError("tokenized training data wasn't generated");
    } else if (net.InputSize() != wordMap.size()) {
        HandleFatalError("number of tokenizer words doesn't match net input count");
    } else if (net.InputSize()*2 != net.OutputSize()) {
        HandleFatalError("Word2Vec net must have twice as many outputs than inputs");
    //} else if (!StrEndsWith(GLOBALS::config_map["LOSS_FUNC"], "CROSS_ENTROPY")) {
    //    HandleFatalError("Word2Vec net must use cross-entropy loss function");
    //} else if (net.LayerActFunc(0) != "SOFTMAX") {
    //    HandleFatalError("Word2Vec input layer must use softmax activation function");
    }

    auto rd = std::random_device{};
    auto rng = std::default_random_engine{ rd() };

    uint32_t trainStep = 0;
    uint32_t inputIndex = 0;
    uint32_t lastInIndex = 0;
    uint32_t targetIndex1 = 0;
    uint32_t targetIndex2 = 0;
    uint32_t lastOut1Index = 0;
    uint32_t lastOut2Index = 0;

    std::vector<GLfloat> targets(net.OutputSize(), GLOBALS::NEG_LABEL);
    std::vector<GLfloat> zeros(net.OutputSize(), 0.0f);
    std::vector<GLfloat> ones(net.OutputSize(), 1.0f);
    std::vector<GLfloat> errors(net.OutputSize(), 0.0f);
    std::vector<GLuint> batchIndices(tokenBatches.size());
    std::iota(batchIndices.begin(), batchIndices.end(), 0);

    openGL.AllocInputSSBO(net.InputSize(), zeros.data());
    openGL.AllocTargetSSBO(net.OutputSize(), targets.data());
    openGL.AllocErrorSSBO(net.OutputSize(), zeros.data());

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
            GLuint exampleNum = exampleMax - 1;
            GLuint percDoneInt = 0;
            GLfloat batchError = 0.0f;

            if (tokens.size() < 3) {
                std::cout << "Skipping batch " << (b+1) << " (too small) " << tokens.size() << std::endl;
                batchCount--;
                continue;
            }

            std::cout << "Training on batch " << (b+1) << " with " << tokens.size() << " tokens" << std::endl;

            for (GLuint i=1; i < exampleMax; ++i)
            {
                inputIndex = tokens[i];
                targetIndex1 = tokens[i+1];
                targetIndex2 = net.InputSize() + tokens[i-1];

                openGL.CopyIndexInToSSBO(inputIndex, lastInIndex, ones);
                openGL.ForwardProp(net, GLOBALS::WORKGROUP_SIZE);

                openGL.CopyIndexOutsToSSBO(targetIndex1, targetIndex2, lastOut1Index, lastOut2Index);
                openGL.BackwardProp(net, GLOBALS::WORKGROUP_SIZE);
                //openGL.BackwardPropFast(net, errors, targetIndexes, GLOBALS::WORKGROUP_SIZE);

                lastInIndex = inputIndex;
                lastOut1Index = targetIndex1;
                lastOut2Index = targetIndex2;

                if (useTrainSteps) {
                    if (++trainStep >= train_steps) {
                        openGL.UpdateWeights(net, GLOBALS::WORKGROUP_SIZE);
                        openGL.UpdateNeurons(net, GLOBALS::WORKGROUP_SIZE);
                        trainStep = 0;
                    }
                } else {
                    openGL.UpdateWeights(net, GLOBALS::WORKGROUP_SIZE);
                }

                percDoneInt = ((float)i / exampleNum) * 100.0f;

                if (percDoneInt < 10) {
                    std::cout << "Progress: " << percDoneInt << "% \r" << std::flush;
                } else if (percDoneInt < 100) {
                    std::cout << "Progress: " << percDoneInt << "%\r" << std::flush;
                } else {
                    std::cout << "Progress: 100%" << std::endl;
                }
            }

            openGL.CopyErrorsFromSSBO(errors);

            for (GLuint i=0; i < errors.size(); ++i)
                batchError += errors[i] / exampleNum;

            batchError /= errors.size();
            epochError += batchError;
            std::cout << "Average batch error: " << batchError << std::endl;

            openGL.ResetErrorSSBO();

            if (net.MemBlendFunc() != "0")
                openGL.ResetNetMem(net, GLOBALS::WORKGROUP_SIZE);
        }

        epochError /= batchCount;
        std::cout << "Average epoch error: " << epochError << std::endl;
    }
}
