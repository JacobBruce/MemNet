#pragma once
#include <random>
#include "OpenGL.h"
#include "Worderizer.h"

inline const uint32_t NET_VERSION = 1;

#pragma pack(push,1)

struct Neuron // 48 bytes
{
	GLfloat bias; // bias weight
	GLfloat bgrad; // bias gradient
	GLfloat mweight; // mem weight
	GLfloat mgrad; // mem gradient
	GLfloat mprev; // previous mem
	GLfloat mem; // memory cell
	GLfloat frate; // forget rate
	GLfloat param; // unused
	GLfloat temp; // used as cache
	GLfloat grad; // error gradient
	GLfloat insum; // input sum
	GLfloat actout; // output val

	Neuron(){}

	Neuron(GLfloat mem_weight)
	{
        bias = 1.0f;
        bgrad = 0.0f;
        mweight = mem_weight;
        mgrad = 0.0f;
        mprev = 0.0f;
        mem = 0.0f;
        frate = 0.5f;
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

	MemNet(const std::vector<GLuint>& net_shape, std::string net_type, GLfloat min_weight, GLfloat max_weight,
           std::string in_act_func, std::string hid_act_func, std::string out_act_func, std::string mem_blend_func)
	{
	    Generate(net_shape, net_type, min_weight, max_weight, in_act_func, hid_act_func, out_act_func, mem_blend_func);
	}

	/*void ConfigLearning(GLfloat learn_rate, GLfloat momentum_strength, GLuint train_steps)
	{
	    learnRate = learn_rate;
	    momentum = momentum_strength;
	    trainSteps = train_steps;
	    gradDiv = train_steps > 0 ? 1.0f / train_steps : 1.0f;
	}*/

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

	    for (GLuint i=1; i < shape.size(); ++i)
        {
            offset += shape[i-1];
            neuronOffsets[i] = offset;
            maxLayerSize = std::max(maxLayerSize, shape[i]);
        }

        offset = shape[0] * shape[1];
	    weightOffsets.resize(shape.size());
	    weightOffsets[0] = 0;
	    weightOffsets[1] = 0;

	    for (GLuint i=2; i < shape.size(); ++i)
        {
            weightOffsets[i] = offset;
            offset += shape[i-1] * shape[i];
        }
	}

    void ApplyFuncStrings()
    {
        std::string* funcStr;
        std::string* derivStr;
        std::string* actFunc;

        if (GLOBALS::config_map["LOSS_FUNC"] == "CROSS_ENTROPY") {
            //GLOBALS::config_map["LOSS_F"] = "-a * log(max(b,3.4e-38))";
            //GLOBALS::config_map["LOSS_D"] = "-a / max(b,3.4e-38)";
            GLOBALS::config_map["LOSS_F"] = "b > 0 ? -a * log(b) : a";
            GLOBALS::config_map["LOSS_D"] = "b > 0 ? -a / b : sign(a)";
        } else if (GLOBALS::config_map["LOSS_FUNC"] == "MAE") {
            GLOBALS::config_map["LOSS_F"] = "abs(a - b)";
            GLOBALS::config_map["LOSS_D"] = "sign(a - b)";
        } else { //MSE
            GLOBALS::config_map["LOSS_F"] = "pow(a - b, 2.0)";
            GLOBALS::config_map["LOSS_D"] = "(a - b) * 2.0";
        }

        GLOBALS::config_map["IN_ACT_F"] = "";
        GLOBALS::config_map["HID_ACT_F"] = "";
        GLOBALS::config_map["OUT_ACT_F"] = "";
        GLOBALS::config_map["IN_ACT_D"] = "";
        GLOBALS::config_map["HID_ACT_D"] = "";
        GLOBALS::config_map["OUT_ACT_D"] = "";

        for (int i=0; i < 3; ++i)
        {
            if (i == 0) {
                funcStr = &GLOBALS::config_map["IN_ACT_F"];
                derivStr = &GLOBALS::config_map["IN_ACT_D"];
                actFunc = &inActFunc;
            } else if (i == 1) {
                funcStr = &GLOBALS::config_map["HID_ACT_F"];
                derivStr = &GLOBALS::config_map["HID_ACT_D"];
                actFunc = &hidActFunc;
            } else {
                funcStr = &GLOBALS::config_map["OUT_ACT_F"];
                derivStr = &GLOBALS::config_map["OUT_ACT_D"];
                actFunc = &outActFunc;
            }

            if (*actFunc == "GAUSSIAN") {
                *funcStr = "return exp(-(x*x*0.5))";
                *derivStr = "return -y * y";
            } else if (*actFunc == "LOGISTIC") {
                *funcStr = "return 1.0 / (1.0+exp(-x))";
                *derivStr = "return y * (1.0 - y)";
            } else if (*actFunc == "SWISH") {
                *funcStr = "float a = 1.0 / (1.0+exp(-2.0*x));\n";
                *funcStr += "neurons[gl_GlobalInvocationID.x].temp = a;\n";
                *funcStr += "return x * a";
                *derivStr = "float a = neurons[gl_GlobalInvocationID.x].temp;\n";
                *derivStr += "return a + y * (1.0 - a)";
            } else if (*actFunc == "GELU") {
                *funcStr = "float a = 0.5 * (1.0 + tanh(0.7978845608028654*(x+pow(0.044715*x,3.0))));\n";
                *funcStr += "neurons[gl_GlobalInvocationID.x].temp = a;\n";
                *funcStr += "return x * a";
                *derivStr = "float x = neurons[gl_GlobalInvocationID.x].insum;\n";
                *derivStr += "return neurons[gl_GlobalInvocationID.x].temp + ";
                *derivStr += "(x * 0.3989422804014327 * exp(pow(x,2.0)*-0.5))";
            } else if (*actFunc == "TANH") {
                *funcStr = "return tanh(x)";
                *derivStr = "return 1.0 - (y * y)";
            } else {
                *funcStr = "return x";
                *derivStr = "return 1.0";
            }
        }
    }

	void Generate(const std::vector<GLuint>& net_shape, std::string net_type, GLfloat min_weight, GLfloat max_weight,
                  std::string in_act_func, std::string hid_act_func, std::string out_act_func, std::string mem_blend_func)
	{
        std::cout << "Generating new net ..." << std::endl;

	    uint64_t weightsPerNeuron = net_shape[0];
	    uint64_t neuronCount = net_shape[0];
	    uint64_t weightCount = 0;

	    version = NET_VERSION;
	    netType = net_type;
        inActFunc = in_act_func;
        hidActFunc = hid_act_func;
        outActFunc = out_act_func;
        memBlendFunc = mem_blend_func;

	    shape.resize(net_shape.size());
	    shape[0] = net_shape[0];

        for (GLuint i=1; i < net_shape.size(); ++i)
        {
            shape[i] = net_shape[i];
            neuronCount += net_shape[i];
            weightCount += net_shape[i] * weightsPerNeuron;
            weightsPerNeuron = net_shape[i];
        }

        weightsPerNeuron = net_shape[0];
        neurons.reserve(neuronCount);
        weights.reserve(weightCount);
        gradients.assign(weightCount, 0.0);

        std::random_device dev;
        std::mt19937 rng(dev());
        std::uniform_real_distribution<float> randWeight(min_weight, max_weight);

        for (GLuint n=0; n < net_shape[0]; ++n)
            neurons.emplace_back(randWeight(rng));

        for (GLuint i=1; i < net_shape.size(); ++i)
        {
            for (GLuint n=0; n < net_shape[i]; ++n)
            {
                neurons.emplace_back(randWeight(rng));

                for (uint64_t w=0; w < weightsPerNeuron; ++w)
                    weights.push_back(randWeight(rng));
            }

            weightsPerNeuron = net_shape[i];
        }

        CalcOffsets();
        CountWeights();
        ApplyFuncStrings();
	}

	void Load(std::string& dir, std::string& name)
	{
	    std::string fileName(dir+name+".config");
	    std::unordered_map<std::string,std::string> netConfig;

	    if (!LoadConfigFile(fileName, netConfig))
            HandleFatalError("failed to open "+fileName);

        version = stoul(netConfig["version"]);
        netType = netConfig["netType"];
        inActFunc = netConfig["inActFunc"];
        hidActFunc = netConfig["hidActFunc"];
        outActFunc = netConfig["outActFunc"];
        memBlendFunc = netConfig["memBlendFunc"];

        GLOBALS::config_map["NET_TYPE"] = netType;
        GLOBALS::config_map["MEM_BLEND_FUNC"] = memBlendFunc;

        std::vector<std::string> shapeVals(ExplodeStr(netConfig["shape"],","));
        shape.resize(shapeVals.size());
        std::cout << "Loading net with shape:";

        for (GLuint i=0; i < shapeVals.size(); ++i)
        {
            shape[i] = stoul(shapeVals[i]);
            std::cout << " " << shapeVals[i];
        }
        std::cout << std::endl;

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
        fread(weights.data(), sizeof(GLfloat), weights.size(), pFile);
        fclose(pFile);

	    fileName = dir+name+".grads";
        pFile = fopen(fileName.c_str(), "rb");
        if (pFile == NULL) HandleFatalError("failed to open "+fileName);
        gradients.resize(stoul(netConfig["weights"]));
        fread(gradients.data(), sizeof(GLfloat), gradients.size(), pFile);
        fclose(pFile);

        CalcOffsets();
        CountWeights();
        ApplyFuncStrings();
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
        fwrite(weights.data(), sizeof(GLfloat), weights.size(), pFile);
        fclose(pFile);

        fileName = dir+name+".grads";
        pFile = fopen(fileName.c_str(), "wb");
        if (pFile == NULL) HandleFatalError("failed to create "+fileName);
        fwrite(gradients.data(), sizeof(GLfloat), gradients.size(), pFile);
        fclose(pFile);

        std::stringstream configSS;
        configSS << "version=" << version << "\n";
        configSS << "neurons=" << neurons.size() << "\n";
        configSS << "weights=" << weights.size() << "\n";
        configSS << "netType=" << netType << "\n";
        configSS << "inActFunc=" << inActFunc << "\n";
        configSS << "hidActFunc=" << hidActFunc << "\n";
        configSS << "outActFunc=" << outActFunc << "\n";
        configSS << "memBlendFunc=" << memBlendFunc << "\n";
        configSS << "shape=";

        for (GLuint i=0; i < shape.size(); ++i)
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

	const std::vector<GLuint>& Shape()
	{
	    return shape;
	}

	const std::vector<Neuron>& Neurons() const
	{
        return neurons;
	}

	const std::vector<GLfloat>& Weights() const
	{
        return weights;
	}

	const std::vector<GLfloat>& Gradients() const
	{
        return gradients;
	}

    Neuron* LayerNeurons(GLuint layer_index)
	{
        return &neurons[NeuronOffset(layer_index)];
	}

    GLfloat* LayerWeights(GLuint layer_index)
	{
        return &weights[WeightOffset(layer_index)];
	}

    GLfloat* LayerGradients(GLuint layer_index)
	{
        return &gradients[WeightOffset(layer_index)];
	}

	GLuint MaxLayerSize() const
	{
	    return maxLayerSize;
	}

	GLuint LayerSize(GLuint layer_index) const
	{
	    return shape[layer_index];
	}

	GLuint LayerCount() const
	{
	    return shape.size();
	}

	GLuint HiddenLayerCount() const
	{
	    return shape.size() - 2;
	}

	GLuint WeightCount(GLuint layer_index) const
	{
	    return weightCounts[layer_index];
	}

	GLuint InputSize() const
	{
	    return shape[0];
	}

	GLuint OutputSize() const
	{
	    return shape[shape.size()-1];
	}

	GLuint OutputOffset() const
	{
        return neurons.size() - OutputSize();
    }

    uint64_t NeuronOffset(GLuint layer_index) const
    {
        return neuronOffsets[layer_index];
    }

    uint64_t WeightOffset(GLuint layer_index) const
    {
        return weightOffsets[layer_index];
    }

    /*const GLfloat& LearnRate() const
    {
        return learnRate;
    }

    const GLfloat& Momentum() const
    {
        return momentum;
    }

    const GLuint& TrainSteps() const
    {
        return trainSteps;
    }*/

    const std::string& MemBlendFunc() const
    {
        return memBlendFunc;
    }

    const std::string& InputActFunc() const
    {
        return inActFunc;
    }

    const std::string& HiddenActFunc() const
    {
        return hidActFunc;
    }

    const std::string& OutputActFunc() const
    {
        return outActFunc;
    }

    const std::string& NetType() const
    {
        return netType;
    }

    const GLuint& NetVersion() const
    {
        return version;
    }

private:
    GLuint version;
    GLuint maxLayerSize;
    //GLuint trainSteps;
	//GLfloat learnRate;
    //GLfloat momentum;
    //GLfloat gradDiv;

    std::string netType;
    std::string inActFunc;
    std::string hidActFunc;
    std::string outActFunc;
    std::string memBlendFunc;

    std::vector<Neuron> neurons;
    std::vector<GLfloat> weights;
    std::vector<GLfloat> gradients;
    std::vector<GLuint> shape;
    std::vector<GLuint> weightCounts;
    std::vector<uint64_t> neuronOffsets;
    std::vector<uint64_t> weightOffsets;
};
