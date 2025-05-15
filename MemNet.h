#pragma once
#include <random>
#include "Resource.h"
#include "Worderizer.h"

inline const uint32_t NET_VERSION = 2;

#pragma pack(push,1)

struct Neuron // 48 bytes
{
	float bias; // bias weight
	float bgrad; // bias gradient
	float mweight; // mem weight
	float mgrad; // mem gradient
	float mprev; // previous mem
	float mem; // memory cell
	float frate; // forget rate
	float param; // unused
	float temp; // used as cache
	float grad; // error gradient
	float insum; // input sum
	float actout; // output val

	Neuron(){}

	Neuron(const float& mem_weight, const float& init_bias, const float& init_frate)
	{
        bias = init_bias;
        bgrad = 0.0f;
        mweight = mem_weight;
        mgrad = 0.0f;
        mprev = 0.0f;
        mem = 0.0f;
        frate = init_frate;
        param = 0.0f;
        temp = 0.0f;
        grad = 0.0f;
        insum = 0.0f;
        actout = 0.0f;
	}
};

#pragma pack(pop)

class MemNet {
public:

	MemNet() {}

	MemNet(std::string net_dir, std::string net_name)
	{
        Load(net_dir, net_name);
	}

	MemNet(const std::vector<uint32_t>& net_shape, std::string net_type, float4 init_data,
           std::vector<std::string>& act_funcs, std::vector<std::string>& mem_layers, std::string mem_blend_func)
	{
	    Generate(net_shape, net_type, init_data, act_funcs, mem_layers, mem_blend_func);
	}

	void CountWeights()
	{
	    weightCounts.resize(shape.size());
	    weightCounts[0] = 0;

	    for (uint32_t i=1; i < shape.size(); ++i)
        {
            uint64_t weightCount = uint64_t(shape[i-1]) * shape[i];

            if (weightCount > UINT_MAX)
                HandleFatalError("connections between layers cannot exceed UINT_MAX");

            weightCounts[i] = weightCount;
        }
    }

	void CalcOffsets()
	{
        uint64_t offset = 0;
	    neuronOffsets.resize(shape.size());
	    neuronOffsets[0] = 0;
	    maxLayerSize = shape[0];

	    for (uint32_t i=1; i < shape.size(); ++i)
        {
            offset += shape[i-1];
            neuronOffsets[i] = offset;
            maxLayerSize = std::max(maxLayerSize, shape[i]);
        }

        offset = shape[0] * shape[1];
	    weightOffsets.resize(shape.size());
	    weightOffsets[0] = 0;
	    weightOffsets[1] = 0;

	    for (uint32_t i=2; i < shape.size(); ++i)
        {
            weightOffsets[i] = offset;
            offset += shape[i-1] * shape[i];
        }
	}

	void SetOutputDims()
	{
        if (netType == "WORD_2_VEC") {
            outLayerSpan = InputSize();
            outLayerRows = 2;
        } else if (GLOBALS::config_map.find("OUTPUT_SPAN") != GLOBALS::config_map.end()) {
            outLayerSpan = stoul(GLOBALS::config_map["OUTPUT_SPAN"]);
            outLayerRows = stoul(GLOBALS::config_map["OUTPUT_ROWS"]);
        } else {
            outLayerSpan = OutputSize();
            outLayerRows = 1;
        }

        if (OutputSize() != outLayerSpan * outLayerRows)
            HandleFatalError("output dimensions (span*rows) does not match size of output layer");
	}

    void ApplyLossFunc()
    {
        bool useLogits = StrIsTrue(GLOBALS::config_map["USE_LOGITS"]);

        if (StrEndsWith(GLOBALS::config_map["LOSS_FUNC"], "CROSS_ENTROPY")) {

            if (GLOBALS::config_map["LOSS_FUNC"] == "BIN_CROSS_ENTROPY") {

                if (useLogits && (actFuncs.back() == "LOGISTIC" || actFuncs.back() == "SIGMOID")) {
                    GLOBALS::config_map["LOSS_F"] = "temp = neurons[gl_GlobalInvocationID.x].insum;\n";
                    GLOBALS::config_map["LOSS_F"] += "return (max(temp, 0.0) - temp * b) + log(1.0 + exp(-abs(temp)))";
                    GLOBALS::config_map["LOSS_D"] = "return temp <= 0.0 ? (temp / (temp+1.0)) - b : (-1.0 / (temp+1.0)) - b+1.0";
                } else {
                    GLOBALS::config_map["LOSS_F"] = "temp = min(max(a, 1e-7), 0.9999999);\n";
                    GLOBALS::config_map["LOSS_F"] += "return -((b * log(temp)) + (1.0 - b) * log(1.0 - temp))";
                    GLOBALS::config_map["LOSS_D"] = "return -((b - temp) / (temp * (1.0 - temp)))";
                }

            } else if (GLOBALS::config_map["LOSS_FUNC"] == "CAT_CROSS_ENTROPY") {

                if (actFuncs.back() != "SOFTMAX")
                    HandleFatalError("categorical cross entropy requires a softmax output layer");

                if (!useLogits) HandleFatalError("categorical cross entropy must use logits");

                GLOBALS::config_map["LOSS_F"] = "return -b * log(min(max(a, 1e-7), 0.9999999))";
                GLOBALS::config_map["LOSS_D"] = "return neurons[gl_GlobalInvocationID.x].insum - b";

            } else if (GLOBALS::config_map["LOSS_FUNC"] == "WEIGHTED_CROSS_ENTROPY") {

                if (actFuncs.back() != "LOGISTIC" && actFuncs.back() != "SIGMOID")
                    HandleFatalError("weighted cross entropy requires a sigmoid/logistic output layer");

                if (!useLogits) HandleFatalError("weighted cross entropy must use logits");

                GLOBALS::config_map["LOSS_F"] = "temp = min(max(neurons[gl_GlobalInvocationID.x].insum, -88.72), 88.72);\n";
                GLOBALS::config_map["LOSS_F"] += "return (1.0 - b) * temp + (1.0 + (POS_WEIGHT - 1.0) * b) ";
                GLOBALS::config_map["LOSS_F"] += "* (log(1.0 + exp(-abs(temp))) + max(-temp, 0.0))";
                GLOBALS::config_map["LOSS_D"] = "return temp != 0.0 ? ((1.0-b) * exp(temp) - (b * POS_WEIGHT)) ";
                GLOBALS::config_map["LOSS_D"] += "/ (exp(temp) + 1.0) : -b * (1.0-POS_WEIGHT)";

            /*} else if (GLOBALS::config_map["LOSS_FUNC"] == "MULTI_WEIGHTED_CROSS_ENTROPY") {

                if (actFuncs.back() != "LOGISTIC" && actFuncs.back() != "SIGMOID")
                    HandleFatalError("weighted cross entropy requires a sigmoid/logistic output layer");

                if (!useLogits) HandleFatalError("weighted cross entropy must use logits");

                GLOBALS::config_map["LOSS_F"] = "temp = neurons[gl_GlobalInvocationID.x].insum;\n";
                GLOBALS::config_map["LOSS_F"] += "return -b * log(min(max(a, 1e-7), 0.9999999))";
                GLOBALS::config_map["LOSS_D"] = "return -((POS_WEIGHT * b * log(temp+1e-7)) + ";
                GLOBALS::config_map["LOSS_D"] += "(NEG_WEIGHT * (1.0 - b) * log(1.0-temp + 1e-7)))";*/
            }

        } else if (GLOBALS::config_map["LOSS_FUNC"] == "BIN_FOCAL_LOSS") {

            if (useLogits) HandleFatalError("cannot use logits with binary focal loss");

            GLOBALS::config_map["LOSS_F"] = "temp = max((b*a) + (1.0-b) * (1.0-a), 1e-7);\n";
            GLOBALS::config_map["LOSS_F"] += "return -((b*ALPHA) + ((1.0-b) * (1.0-ALPHA))) ";
            GLOBALS::config_map["LOSS_F"] += "* pow(1.0-temp, GAMMA) * log(temp)";
            GLOBALS::config_map["LOSS_D"] = "return b * pow(1.0 - temp, GAMMA) * ";
            GLOBALS::config_map["LOSS_D"] += "(GAMMA * temp * log(temp) + temp - 1.0)";

        } else {

            if (useLogits) HandleFatalError("cannot use logits with MAE or MSE loss functions");

            if (GLOBALS::config_map["LOSS_FUNC"] == "MAE") {
                GLOBALS::config_map["LOSS_F"] = "return abs(a - b)";
                GLOBALS::config_map["LOSS_D"] = "return sign(a - b)";
            } else { //MSE
                GLOBALS::config_map["LOSS_F"] = "return pow(a - b, 2.0)";
                GLOBALS::config_map["LOSS_D"] = "return (a - b) * 2.0";
            }
        }
    }

    static void ApplyActFunc(std::string act_func)
    {
        std::string& funcStr(GLOBALS::config_map["ACT_F"]);
        std::string& derivStr(GLOBALS::config_map["ACT_D"]);
        std::string& derivOutStr(GLOBALS::config_map["ACT_D_OUT"]);

        if (act_func == "GAUSSIAN") {
            funcStr = "return exp(-(x*x*0.5))";
            derivStr = "return -y * exp(-0.5*y*y)";
        } else if (act_func == "SOFTMAX") {
            funcStr = "return exp(min(x, 88.72))"; // divided by sum later
            derivStr = "return 1.0"; // combined with loss function deriv
        } else if (act_func == "LOGISTIC" || act_func == "SIGMOID") {
            funcStr = "return 1.0 / (1.0+exp(-max(x, -88.72)))";
            derivStr = "return y * (1.0 - y)";
        } else if (act_func == "SWISH") {
            funcStr = "float a = 1.0 / (1.0+exp(-2.0*x));\n";
            funcStr += "neurons[gl_GlobalInvocationID.x].temp = a;\n";
            funcStr += "return x * a";
            derivStr = "float a = neurons[gl_GlobalInvocationID.x].temp;\n";
            derivStr += "return a + y * (1.0 - a)";
        } else if (act_func == "GELU") {
            funcStr = "float a = 0.5 * (1.0 + tanh(0.7978845608028654*(x+pow(0.044715*x,3.0))));\n";
            funcStr += "neurons[gl_GlobalInvocationID.x].temp = a;\n";
            funcStr += "return x * a";
            derivStr = "float x = neurons[gl_GlobalInvocationID.x].insum;\n";
            derivStr += "return neurons[gl_GlobalInvocationID.x].temp + ";
            derivStr += "(x * 0.3989422804014327 * exp(pow(x,2.0)*-0.5))";
        } else if (act_func == "TANH") {
            funcStr = "return tanh(x)";
            derivStr = "return 1.0 - (y * y)";
        } else {
            funcStr = "return x";
            derivStr = "return 1.0";
        }

        if (StrIsTrue(GLOBALS::config_map["USE_LOGITS"])) {
            derivOutStr = "return 1.0";
        } else {
            derivOutStr = derivStr;
        }
    }

	void Generate(const std::vector<uint32_t>& net_shape, std::string net_type, float4 init_data,
                  const std::vector<std::string>& act_funcs, std::vector<std::string>& mem_layers, std::string mem_blend_func)
	{
        std::cout << "Generating new net ..." << std::endl;

        if (net_shape.size() != act_funcs.size())
            HandleFatalError("number of activation functions does not match number of layers in net");

	    uint64_t weightsPerNeuron = net_shape[0];
	    uint64_t neuronCount = net_shape[0];
	    uint64_t weightCount = 0;

	    version = NET_VERSION;
	    netType = net_type;
        actFuncs = act_funcs;
        memBlendFunc = mem_blend_func;

        memBlendBools.resize(net_shape.size());
        softmaxBools.resize(net_shape.size());

        for (uint32_t i=0; i < net_shape.size(); ++i)
        {
            if (StrIsFalse(mem_layers[i])) {
                memBlendBools[i] = false;
            } else {
                memBlendBools[i] = true;
            }

            if (actFuncs[i] == "SOFTMAX") {
                if (i != net_shape.size()-1)
                    HandleFatalError("softmax can only be used on output layer");
                softmaxBools[i] = true;
            } else {
                softmaxBools[i] = false;
            }
        }

	    shape.resize(net_shape.size());
	    shape[0] = net_shape[0];

        for (uint32_t i=1; i < net_shape.size(); ++i)
        {
            shape[i] = net_shape[i];
            neuronCount += net_shape[i];
            weightCount += net_shape[i] * weightsPerNeuron;
            weightsPerNeuron = net_shape[i];
        }

        weightsPerNeuron = net_shape[0];
        neurons.reserve(neuronCount);
        weights.reserve(weightCount);
        gradients.assign(weightCount, 0.f);

        std::random_device dev;
        std::mt19937 rng(dev());
        std::uniform_real_distribution<float> randWeight(init_data.x, init_data.y);

        for (uint32_t n=0; n < net_shape[0]; ++n)
            neurons.emplace_back(randWeight(rng), init_data.z, init_data.w);

        for (uint32_t i=1; i < net_shape.size(); ++i)
        {
            for (uint32_t n=0; n < net_shape[i]; ++n)
            {
                neurons.emplace_back(randWeight(rng), init_data.z, init_data.w);

                for (uint64_t w=0; w < weightsPerNeuron; ++w)
                    weights.emplace_back(randWeight(rng));
            }

            weightsPerNeuron = net_shape[i];
        }

        CalcOffsets();
        SetOutputDims();
        CountWeights();
        ApplyLossFunc();
	}

	void Load(std::string& dir, std::string& name)
	{
	    std::string fileName(dir+name+".config");
	    std::unordered_map<std::string,std::string> netConfig;

	    LoadConfigFile(fileName, netConfig);

        version = stoul(netConfig["version"]);

        if (version != NET_VERSION)
            HandleFatalError("can't load net made for different version of MemNet");

        netType = netConfig["netType"];
        memBlendFunc = netConfig["memBlendFunc"];

        SplitStr(netConfig["actFuncs"], ",", actFuncs);

        std::vector<std::string> strParts;
        SplitStr(netConfig["memLayers"], ",", strParts);
        memBlendBools.resize(strParts.size());
        softmaxBools.resize(actFuncs.size());

        if (strParts.size() != actFuncs.size())
            HandleFatalError("net config file is corrupted!");

        for (uint32_t i=0; i < strParts.size(); ++i)
        {
            if (StrIsFalse(strParts[i])) {
                memBlendBools[i] = false;
            } else {
                memBlendBools[i] = true;
            }

            softmaxBools[i] = (actFuncs[i] == "SOFTMAX");
        }

        GLOBALS::config_map["NET_TYPE"] = netType;
        GLOBALS::config_map["MEM_BLEND_FUNC"] = memBlendFunc;

        std::vector<std::string> shapeVals(ExplodeStr(netConfig["shape"],","));
        shape.resize(shapeVals.size());
        std::cout << "Loading net with shape:";

        for (uint32_t i=0; i < shapeVals.size(); ++i)
        {
            shape[i] = stoul(shapeVals[i]);
            std::cout << " " << shapeVals[i];
        }
        std::cout << std::endl;

        outLayerSpan = stoul(netConfig["outputSpan"]);
        outLayerRows = shape.back() / outLayerSpan;

	    fileName = dir+name+".neurons";
        FILE* pFile = fopen(fileName.c_str(), "rb");
        if (pFile == NULL) HandleFatalError("failed to open "+fileName);
        neurons.resize(stoul(netConfig["neurons"]));
        fread(neurons.data(), sizeof(Neuron), neurons.size(), pFile);
        fclose(pFile);

	    fileName = dir+name+".weights";
        pFile = fopen(fileName.c_str(), "rb");
        if (pFile == NULL) HandleFatalError("failed to open "+fileName);
        weights.resize(stoul(netConfig["weights"]));
        fread(weights.data(), sizeof(float), weights.size(), pFile);
        fclose(pFile);

	    fileName = dir+name+".grads";

	    if (FileExists(fileName)) {
            pFile = fopen(fileName.c_str(), "rb");
            if (pFile == NULL) HandleFatalError("failed to open "+fileName);
            gradients.resize(stoul(netConfig["weights"]));
            fread(gradients.data(), sizeof(float), gradients.size(), pFile);
            fclose(pFile);
	    } else if (GLOBALS::config_map["ENGINE_MODE"] == "1") {
            std::cout << "WARNING: model has no gradient checkpoint file" << std::endl;
            gradients.assign(stoul(netConfig["weights"]), 0.f);
	    }

        CalcOffsets();
        CountWeights();
        ApplyLossFunc();
	}

	void Save(std::string& dir, std::string& name)
	{
        std::cout << "Saving net ...";

	    std::string fileName(dir+name+".neurons");
        FILE* pFile = fopen(fileName.c_str(), "wb");
        if (pFile == NULL) HandleFatalError("failed to create "+fileName);
        fwrite(neurons.data(), sizeof(Neuron), neurons.size(), pFile);
        fclose(pFile);

        fileName = dir+name+".weights";
        pFile = fopen(fileName.c_str(), "wb");
        if (pFile == NULL) HandleFatalError("failed to create "+fileName);
        fwrite(weights.data(), sizeof(float), weights.size(), pFile);
        fclose(pFile);

        if (!gradients.empty()) {
            fileName = dir+name+".grads";
            pFile = fopen(fileName.c_str(), "wb");
            if (pFile == NULL) HandleFatalError("failed to create "+fileName);
            fwrite(gradients.data(), sizeof(float), gradients.size(), pFile);
            fclose(pFile);
        }

        std::stringstream configSS;
        configSS << "version=" << version << "\n";
        configSS << "neurons=" << neurons.size() << "\n";
        configSS << "weights=" << weights.size() << "\n";
        configSS << "netType=" << netType << "\n";
        configSS << "outputSpan=" << outLayerSpan << "\n";
        configSS << "memBlendFunc=" << memBlendFunc << "\n";
        configSS << "memLayers=";

        for (uint32_t i=0; i < memBlendBools.size(); ++i)
        {
            configSS << memBlendBools[i];
            if (i < memBlendBools.size()-1) {
                configSS << ",";
            } else {
                configSS << "\n";
            }
        }

        configSS << "actFuncs=";

        for (uint32_t i=0; i < actFuncs.size(); ++i)
        {
            configSS << actFuncs[i];
            if (i < actFuncs.size()-1) {
                configSS << ",";
            } else {
                configSS << "\n";
            }
        }

        configSS << "shape=";

        for (uint32_t i=0; i < shape.size(); ++i)
        {
            configSS << shape[i];
            if (i < shape.size()-1) {
                configSS << ",";
            } else {
                configSS << "\n";
            }
        }

        fileName = dir+name+".config";
        std::ofstream configFile(fileName);
        if (!configFile.is_open()) HandleFatalError("failed to create "+fileName);
        configFile << configSS.str();
        configFile.close();
	}

	const std::vector<uint32_t>& Shape()
	{
	    return shape;
	}

	const std::vector<Neuron>& Neurons() const
	{
        return neurons;
	}

	const std::vector<float>& Weights() const
	{
        return weights;
	}

	const std::vector<float>& Gradients() const
	{
        return gradients;
	}

    Neuron* LayerNeurons(uint32_t layer_index)
	{
        return &neurons[NeuronOffset(layer_index)];
	}

	Neuron* InputNeurons()
	{
        return LayerNeurons(0);
    }

	Neuron* OutputNeurons()
	{
        return LayerNeurons(LayerCount()-1);
    }

    float* LayerWeights(uint32_t layer_index)
	{
        return &weights[WeightOffset(layer_index)];
	}

    float* LayerGradients(uint32_t layer_index)
	{
        return &gradients[WeightOffset(layer_index)];
	}

	const uint32_t& MaxLayerSize() const
	{
	    return maxLayerSize;
	}

	const uint32_t& LayerSize(uint32_t layer_index) const
	{
	    return shape[layer_index];
	}

	uint32_t LayerCount() const
	{
	    return shape.size();
	}

	uint32_t HiddenLayerCount() const
	{
	    return shape.size() - 2;
	}

	const uint32_t& WeightCount(uint32_t layer_index) const
	{
	    return weightCounts[layer_index];
	}

	const uint32_t& InputSize() const
	{
	    return shape[0];
	}

	const uint32_t& OutputSize() const
	{
	    return shape[shape.size()-1];
	}

	uint32_t OutputOffset() const
	{
        return neurons.size() - OutputSize();
    }

	const uint32_t& OutputSpan() const
	{
        return outLayerSpan;
    }

	const uint32_t& OutputRows() const
	{
        return outLayerRows;
    }

    const uint64_t& NeuronOffset(uint32_t layer_index) const
    {
        return neuronOffsets[layer_index];
    }

    const uint64_t& WeightOffset(uint32_t layer_index) const
    {
        return weightOffsets[layer_index];
    }

    bool LayerUsesSoftMax(uint32_t layer_index) const
    {
        return softmaxBools[layer_index];
    }

    bool LayerHasMem(uint32_t layer_index) const
    {
        return memBlendBools[layer_index];
    }

    const std::string& MemBlendFunc() const
    {
        return memBlendFunc;
    }

    const std::string& LayerActFunc(uint32_t layer_index) const
    {
        return actFuncs[layer_index];
    }

    const std::vector<std::string>& ActivationFuncs() const
    {
        return actFuncs;
    }

    const std::string& OutputActFunc() const
    {
        return actFuncs.back();
    }

    const std::string& NetType() const
    {
        return netType;
    }

    const uint32_t& NetVersion() const
    {
        return version;
    }

private:
    uint32_t version;
    uint32_t maxLayerSize;
    uint32_t outLayerSpan;
    uint32_t outLayerRows;

    std::string netType;
    std::string memBlendFunc;
    std::vector<bool> memBlendBools;
    std::vector<bool> softmaxBools;
    std::vector<std::string> actFuncs;
    std::vector<Neuron> neurons;
    std::vector<float> weights;
    std::vector<float> gradients;
    std::vector<uint32_t> shape;
    std::vector<uint32_t> weightCounts;
    std::vector<uint64_t> neuronOffsets;
    std::vector<uint64_t> weightOffsets;
};
