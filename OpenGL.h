#pragma once
#include <string>
#include <gl/glew.h>
#include <GLFW/glfw3.h>
#include "ReadWrite.h"
#include "Resource.h"
#include "MemNet.h"

inline const std::string glslNeuronStr =
"struct Neuron {\n"
"	float bias;\n"
"	float bgrad;\n"
"	float mweight;\n"
"	float mgrad;\n"
"	float mprev;\n"
"	float mem;\n"
"	float frate;\n"
"	float param;\n"
"	float temp;\n"
"	float grad;\n"
"	float insum;\n"
"	float actout;\n"
"}";

class ShaderSource
{
private:
    const char* srcPtr;
public:
    std::string shaderFile;
    std::string shaderSource;
    GLuint& shaderID;
    GLuint& programID;

    ShaderSource(const std::string& source_file, GLuint& shader_id, GLuint& program_id) :
        shaderFile(source_file),
        shaderSource(ReadFileStr(source_file)),
        shaderID(shader_id),
        programID(program_id)
    {
        uint32_t trainSteps = stoul(GLOBALS::config_map["TRAIN_STEPS"]);
        float gradDiv = trainSteps > 0 ? 1.0f / trainSteps : 1.0f;

        ReplaceSubstr(shaderSource, "NEURON_STRUCT", glslNeuronStr);
        ReplaceSubstr(shaderSource, "GRAD_DIV", std::to_string(gradDiv));
        ReplaceSubstr(shaderSource, "WORKGROUP_SIZE", std::to_string(GLOBALS::WORKGROUP_SIZE));
        ReplaceSubstr(shaderSource, "LEARN_RATE_BIAS", GLOBALS::config_map["LEARN_RATE_BIAS"]);
        ReplaceSubstr(shaderSource, "LEARN_RATE_MEM", GLOBALS::config_map["LEARN_RATE_MEM"]);
        ReplaceSubstr(shaderSource, "LEARN_RATE", GLOBALS::config_map["LEARN_RATE"]);
        ReplaceSubstr(shaderSource, "MOMENTUM", GLOBALS::config_map["MOMENTUM"]);
        ReplaceSubstr(shaderSource, "TRAIN_STEPS", GLOBALS::config_map["TRAIN_STEPS"]);
        ReplaceSubstr(shaderSource, "MEM_BLEND_FUNC", GLOBALS::config_map["MEM_BLEND_FUNC"]);
        ReplaceSubstr(shaderSource, "IN_ACT_FUNC", GLOBALS::config_map["IN_ACT_F"]);
        ReplaceSubstr(shaderSource, "HID_ACT_FUNC", GLOBALS::config_map["HID_ACT_F"]);
        ReplaceSubstr(shaderSource, "OUT_ACT_FUNC", GLOBALS::config_map["OUT_ACT_F"]);
        ReplaceSubstr(shaderSource, "IN_ACT_DERIV", GLOBALS::config_map["IN_ACT_D"]);
        ReplaceSubstr(shaderSource, "HID_ACT_DERIV", GLOBALS::config_map["HID_ACT_D"]);
        ReplaceSubstr(shaderSource, "OUT_ACT_DERIV", GLOBALS::config_map["OUT_ACT_D"]);
        ReplaceSubstr(shaderSource, "LOSS_FUNC", GLOBALS::config_map["LOSS_F"]);
        ReplaceSubstr(shaderSource, "LOSS_DERIV", GLOBALS::config_map["LOSS_D"]);

        srcPtr = shaderSource.c_str();
    }

    const GLchar** GetGLPtr()
    {
        return &srcPtr;
    }
};

class GL
{
protected:
    //TODO: option to use single layer buffers (layerSSBO, weightSSBO)
    std::vector<GLuint> layerSSBOs;
    std::vector<GLuint> weightSSBOs;
    std::vector<GLuint> gradSSBOs;
    //GLuint layerSSBO;
    //GLuint weightSSBO;
    GLuint errorSSBO;
    GLuint inputSSBO;
    GLuint outputSSBO;
    GLuint forwardInProg;
    GLuint forwardInShader;
    GLuint forwardNProg;
    GLuint forwardNShader;
    GLuint forwardOutProg;
    GLuint forwardOutShader;
    GLuint backwardInProg;
    GLuint backwardInShader;
    GLuint backwardOutProg;
    GLuint backwardOutShader;
    GLuint backwardNProg;
    GLuint backwardNShader;
    GLuint fBackS1Prog;
    GLuint fBackS1Shader;
    GLuint fSumGradsProg;
    GLuint fSumGradsShader;
    GLuint updateWeightsProg;
    GLuint updateWeightsShader;
    GLuint updateErrorsProg;
    GLuint updateErrorsShader;
    GLuint updateNeuronProg;
    GLuint updateNeuronShader;
    GLuint resetMemProg;
    GLuint resetMemShader;
    GLuint sumGradsProg;
    GLuint sumGradsShader;
    GLuint activeShader;
    GLuint fnwid, fowid, bnwid,
           bowid, sgwid, sgoid,
           wboid, wsoid, wbwid, wswid;

public:

    GL() {}

    ~GL()
    {
        glDeleteProgram(forwardInProg);
        glDeleteProgram(forwardNProg);
        glDeleteProgram(forwardOutProg);
        glDeleteShader(forwardInShader);
        glDeleteShader(forwardNShader);
        glDeleteShader(forwardOutShader);

        glDeleteProgram(backwardInProg);
        glDeleteProgram(backwardOutProg);
        glDeleteProgram(backwardNProg);
        glDeleteShader(backwardInShader);
        glDeleteShader(backwardOutShader);
        glDeleteShader(backwardNShader);

        glDeleteProgram(updateWeightsProg);
        glDeleteProgram(updateErrorsProg);
        glDeleteProgram(updateNeuronProg);
        glDeleteProgram(resetMemProg);
        glDeleteProgram(sumGradsProg);

        glDeleteShader(updateWeightsShader);
        glDeleteShader(updateErrorsShader);
        glDeleteShader(updateNeuronShader);
        glDeleteShader(resetMemShader);
        glDeleteShader(sumGradsShader);

        glDeleteProgram(fBackS1Prog);
        glDeleteProgram(fSumGradsProg);
        glDeleteShader(fBackS1Shader);
        glDeleteShader(fSumGradsShader);

        glDeleteBuffers(layerSSBOs.size(), layerSSBOs.data());
        glDeleteBuffers(weightSSBOs.size(), weightSSBOs.data());
        glDeleteBuffers(gradSSBOs.size(), gradSSBOs.data());
        //glDeleteBuffers(1, &layerSSBO);
        //glDeleteBuffers(1, &weightSSBO);
        glDeleteBuffers(1, &errorSSBO);
        glDeleteBuffers(1, &inputSSBO);
        glDeleteBuffers(1, &outputSSBO);
    }

    void SetupMainBuffers(GLuint layer_count)
    {
        // Create Shader Storage Buffers
        layerSSBOs.resize(layer_count);
        weightSSBOs.resize(layer_count-1);
        gradSSBOs.resize(layer_count-1);

        glGenBuffers(layer_count, layerSSBOs.data());
        glGenBuffers(layer_count-1, weightSSBOs.data());
        glGenBuffers(layer_count-1, gradSSBOs.data());
        //glGenBuffers(1, &layerSSBO);
        //glGenBuffers(1, &weightSSBO);
        glGenBuffers(1, &errorSSBO);
        glGenBuffers(1, &inputSSBO);
        glGenBuffers(1, &outputSSBO);
    }

    void SetupShaders(const bool use_avg_shaders)
    {
        // Read neuron shader source files
        std::vector<ShaderSource> shaders;
        shaders.emplace_back(GLOBALS::DATA_FOLDER+"shaders/forward_in.glsl", forwardInShader, forwardInProg);
        shaders.emplace_back(GLOBALS::DATA_FOLDER+"shaders/forward_n.glsl", forwardNShader, forwardNProg);
        shaders.emplace_back(GLOBALS::DATA_FOLDER+"shaders/forward_out.glsl", forwardOutShader, forwardOutProg);
        shaders.emplace_back(GLOBALS::DATA_FOLDER+"shaders/sum_ngrads.glsl", sumGradsShader, sumGradsProg);
        shaders.emplace_back(GLOBALS::DATA_FOLDER+"shaders/update_errors.glsl", updateErrorsShader, updateErrorsProg);
        shaders.emplace_back(GLOBALS::DATA_FOLDER+"shaders/reset_mem.glsl", resetMemShader, resetMemProg);
        shaders.emplace_back(GLOBALS::DATA_FOLDER+"shaders/fast/backward_s1.glsl", fBackS1Shader, fBackS1Prog);
        shaders.emplace_back(GLOBALS::DATA_FOLDER+"shaders/fast/sum_ngrads.glsl", fSumGradsShader, fSumGradsProg);

        if (use_avg_shaders) {
            shaders.emplace_back(GLOBALS::DATA_FOLDER+"shaders/backward_in_avg.glsl", backwardInShader, backwardInProg);
            shaders.emplace_back(GLOBALS::DATA_FOLDER+"shaders/backward_out_avg.glsl", backwardOutShader, backwardOutProg);
            shaders.emplace_back(GLOBALS::DATA_FOLDER+"shaders/backward_n_avg.glsl", backwardNShader, backwardNProg);
            shaders.emplace_back(GLOBALS::DATA_FOLDER+"shaders/update_weights_avg.glsl", updateWeightsShader, updateWeightsProg);
            shaders.emplace_back(GLOBALS::DATA_FOLDER+"shaders/update_neuron.glsl", updateNeuronShader, updateNeuronProg);
        } else {
            shaders.emplace_back(GLOBALS::DATA_FOLDER+"shaders/backward_in.glsl", backwardInShader, backwardInProg);
            shaders.emplace_back(GLOBALS::DATA_FOLDER+"shaders/backward_out.glsl", backwardOutShader, backwardOutProg);
            shaders.emplace_back(GLOBALS::DATA_FOLDER+"shaders/backward_n.glsl", backwardNShader, backwardNProg);
            shaders.emplace_back(GLOBALS::DATA_FOLDER+"shaders/update_weights.glsl", updateWeightsShader, updateWeightsProg);
        }

        for (GLuint i=0; i < shaders.size(); ++i)
        {
            // Compile shader from source
            shaders[i].shaderID = glCreateShader(GL_COMPUTE_SHADER);
            glShaderSource(shaders[i].shaderID, 1, shaders[i].GetGLPtr(), nullptr);
            glCompileShader(shaders[i].shaderID);
            CheckCompileStatus(shaders[i]);

            // Attach and link shader to shader program
            shaders[i].programID = glCreateProgram();
            glAttachShader(shaders[i].programID, shaders[i].shaderID);
            glLinkProgram(shaders[i].programID);

            // Check shader program is valid
            CheckProgramStatus(shaders[i], GL_LINK_STATUS);
            glValidateProgram(shaders[i].programID);
            CheckProgramStatus(shaders[i], GL_VALIDATE_STATUS);
        }

        // Setup compute shader uniforms
        glUseProgram(forwardNProg);
        fnwid = glGetUniformLocation(forwardNProg, "weightsPerNeuron");

        glUseProgram(forwardOutProg);
        fowid = glGetUniformLocation(forwardOutProg, "weightsPerNeuron");

        glUseProgram(backwardNProg);
        bnwid = glGetUniformLocation(backwardNProg, "weightsPerNeuron");

        glUseProgram(backwardOutProg);
        bowid = glGetUniformLocation(backwardOutProg, "weightsPerNeuron");

        glUseProgram(sumGradsProg);
        sgwid = glGetUniformLocation(sumGradsProg, "weightsPerNeuron");
        sgoid = glGetUniformLocation(sumGradsProg, "otherLayerSize");

        glUseProgram(fBackS1Prog);
        wbwid = glGetUniformLocation(fBackS1Prog, "weightsPerNeuron");
        wboid = glGetUniformLocation(fBackS1Prog, "outIndex");

        glUseProgram(fSumGradsProg);
        wswid = glGetUniformLocation(fSumGradsProg, "weightsPerNeuron");
        wsoid = glGetUniformLocation(fSumGradsProg, "outIndex");
    }

    void Initialize(GLuint layer_count, bool use_avg_shaders)
    {
        // Setup buffers and shaders
        SetupMainBuffers(layer_count);
        SetupShaders(use_avg_shaders);

        glBindBuffer(GL_ARRAY_BUFFER, 0);

        if (PrintErrors()) exit(EXIT_FAILURE);
    }

    inline void BindShader(const GLuint shader_id)
    {
        glUseProgram(shader_id);
        activeShader = shader_id;
    }

    inline void CopyNeuronsToSSBO(MemNet& net, GLuint layer_index)
    {
        GLuint neuronCount = net.LayerSize(layer_index);
        assert(neuronCount > 0);
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, layerSSBOs[layer_index]);
        glBufferData(GL_SHADER_STORAGE_BUFFER, sizeof(Neuron)*neuronCount, net.LayerNeurons(layer_index), GL_DYNAMIC_DRAW);
        glFinish();
    }

    inline void CopyWeightsToSSBO(MemNet& net, GLuint layer_index)
    {
        GLuint weightCount = net.WeightCount(layer_index);
        assert(layer_index > 0 && weightCount > 0);
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, weightSSBOs[layer_index-1]);
        glBufferData(GL_SHADER_STORAGE_BUFFER, sizeof(GLfloat)*weightCount, net.LayerWeights(layer_index), GL_DYNAMIC_DRAW);
        glFinish();
    }

    inline void CopyGradsToSSBO(MemNet& net, GLuint layer_index)
    {
        GLuint gradCount = net.WeightCount(layer_index);
        assert(layer_index > 0 && gradCount > 0);
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, gradSSBOs[layer_index-1]);
        glBufferData(GL_SHADER_STORAGE_BUFFER, sizeof(GLfloat)*gradCount, net.LayerGradients(layer_index), GL_DYNAMIC_DRAW);
        glFinish();
    }

    inline void ResetErrorSSBO()
    {
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, errorSSBO);
        glClearBufferData(GL_SHADER_STORAGE_BUFFER, GL_R32F, GL_RED, GL_FLOAT, NULL);
        glFinish();
    }

    inline void AllocErrorSSBO(GLuint out_layer_size)
    {
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, errorSSBO);
        glBufferData(GL_SHADER_STORAGE_BUFFER, sizeof(GLfloat)*out_layer_size, NULL, GL_DYNAMIC_DRAW);
        glFinish();
        ResetErrorSSBO();
    }

    inline void AllocInputSSBO(GLuint in_layer_size, const void* zero_inputs)
    {
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, inputSSBO);
        glBufferData(GL_SHADER_STORAGE_BUFFER, sizeof(GLfloat)*in_layer_size, zero_inputs, GL_DYNAMIC_DRAW);
        glFinish();
    }

    inline void AllocTargetSSBO(GLuint out_layer_size, const void* zero_outputs)
    {
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, outputSSBO);
        glBufferData(GL_SHADER_STORAGE_BUFFER, sizeof(GLfloat)*out_layer_size, zero_outputs, GL_DYNAMIC_DRAW);
        glFinish();
    }

    inline void CopyIndexInToSSBO(GLuint& input_index, GLuint& last_in_index, std::vector<GLfloat>& inputs)
    {
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, inputSSBO);
        glBufferSubData(GL_SHADER_STORAGE_BUFFER, sizeof(GLfloat)*last_in_index, sizeof(GLfloat), (void*)0);
        glBufferSubData(GL_SHADER_STORAGE_BUFFER, sizeof(GLfloat)*input_index, sizeof(GLfloat), &inputs[input_index]);
        glFinish();
    }

    inline void CopyIndexOutToSSBO(GLuint& target_index, GLuint& last_out_index, std::vector<GLfloat>& targets)
    {
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, outputSSBO);
        glBufferSubData(GL_SHADER_STORAGE_BUFFER, sizeof(GLfloat)*last_out_index, sizeof(GLfloat), (void*)0);
        glBufferSubData(GL_SHADER_STORAGE_BUFFER, sizeof(GLfloat)*target_index, sizeof(GLfloat), &targets[target_index]);
        glFinish();
    }

    /*inline void CopyW2VoutsToSSBO(GLuint& target1_index, GLuint& target2_index, GLuint& last_out1_index,
                                  GLuint& last_out2_index, std::vector<GLfloat>& targets)
    {
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, outputSSBO);
        glBufferSubData(GL_SHADER_STORAGE_BUFFER, sizeof(GLfloat)*last_out1_index, sizeof(GLfloat), (void*)0);
        glBufferSubData(GL_SHADER_STORAGE_BUFFER, sizeof(GLfloat)*last_out2_index, sizeof(GLfloat), (void*)0);
        glBufferSubData(GL_SHADER_STORAGE_BUFFER, sizeof(GLfloat)*target1_index, sizeof(GLfloat), &targets[target1_index]);
        glBufferSubData(GL_SHADER_STORAGE_BUFFER, sizeof(GLfloat)*target2_index, sizeof(GLfloat), &targets[target2_index]);
        glFinish();
    }

    inline void CopyInputToSSBO(const GLuint& input_index, const GLfloat& val)
    {
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, inputSSBO);
        glBufferSubData(GL_SHADER_STORAGE_BUFFER, sizeof(GLfloat)*input_index, sizeof(GLfloat), &val);
        glFinish();
    }

    inline void CopyTargetToSSBO(const GLuint& target_index, const GLfloat& val)
    {
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, outputSSBO);
        glBufferSubData(GL_SHADER_STORAGE_BUFFER, sizeof(GLfloat)*target_index, sizeof(GLfloat), &val);
        glFinish();
    }*/

    inline void CopyInputsToSSBO(const GLuint& inputs_size, const void* inputs)
    {
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, inputSSBO);
        glBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, sizeof(GLfloat)*inputs_size, inputs);
        glFinish();
    }

    inline void CopyTargetsToSSBO(const GLuint& outputs_size, const void* targets)
    {
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, outputSSBO);
        glBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, sizeof(GLfloat)*outputs_size, targets);
        glFinish();
    }

    inline void CopyNetToBuffers(MemNet& net)
    {
        for (GLuint i=0; i < net.LayerCount(); ++i)
            CopyNeuronsToSSBO(net, i);

        for (GLuint i=1; i < net.LayerCount(); ++i)
        {
            CopyWeightsToSSBO(net, i);
            CopyGradsToSSBO(net, i);
        }
    }

    inline void CopyNeuronsFromSSBO(MemNet& net, const GLuint& layer_index)
    {
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, layerSSBOs[layer_index]);

        glGetBufferSubData(
            GL_SHADER_STORAGE_BUFFER, 0,
            sizeof(Neuron)*net.LayerSize(layer_index),
            net.LayerNeurons(layer_index)
        );

        glFinish();
    }

    inline void CopyWeightsFromSSBO(MemNet& net, const GLuint& layer_index)
    {
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, weightSSBOs[layer_index-1]);

        glGetBufferSubData(
            GL_SHADER_STORAGE_BUFFER, 0,
            sizeof(GLfloat)*net.WeightCount(layer_index),
            net.LayerWeights(layer_index)
        );

        glFinish();
    }

    inline void CopyGradsFromSSBO(MemNet& net, const GLuint& layer_index)
    {
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, gradSSBOs[layer_index-1]);

        glGetBufferSubData(
            GL_SHADER_STORAGE_BUFFER, 0,
            sizeof(GLfloat)*net.WeightCount(layer_index),
            net.LayerGradients(layer_index)
        );

        glFinish();
    }

    inline void CopyNetFromBuffers(MemNet& net)
    {
        for (GLuint i=0; i < net.LayerCount(); ++i)
            CopyNeuronsFromSSBO(net, i);

        for (GLuint i=1; i < net.LayerCount(); ++i)
        {
            CopyWeightsFromSSBO(net, i);
            CopyGradsFromSSBO(net, i);
        }
    }

    inline void CopyErrorsFromSSBO(std::vector<GLfloat>& dest)
    {
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, errorSSBO);

        glGetBufferSubData(
            GL_SHADER_STORAGE_BUFFER, 0,
            sizeof(GLfloat)*dest.size(), dest.data()
        );

        glFinish();
    }

    inline void CopyOutputLayer(MemNet& net, std::vector<Neuron>& dest)
    {
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, layerSSBOs[net.LayerCount()-1]);

        glGetBufferSubData(
            GL_SHADER_STORAGE_BUFFER, 0,
            sizeof(Neuron)*dest.size(), dest.data()
        );

        glFinish();
    }

    inline void CompTargNeuronsW2V(std::vector<GLfloat>& errors, const GLuint& layer_index, const GLuint& target_index)
    {
        Neuron targNeuron;

        glBindBuffer(GL_SHADER_STORAGE_BUFFER, layerSSBOs[layer_index]);

        glGetBufferSubData(
            GL_SHADER_STORAGE_BUFFER, target_index*sizeof(Neuron),
            sizeof(Neuron), &targNeuron
        );

        glFinish();

        GLfloat outVal = targNeuron.actout;
        GLfloat outGrad = (1.0f - (outVal * outVal)) * (-1.0f / outVal);

        targNeuron.grad = outGrad;
        //targNeuron.bgrad += outGrad;
        //targNeuron.mgrad += outGrad * targNeuron.mprev;

        glBufferSubData(
            GL_SHADER_STORAGE_BUFFER, target_index*sizeof(Neuron),
            sizeof(Neuron), &targNeuron
        );

        errors[target_index] += targNeuron.actout > 0.0f ? -std::log(targNeuron.actout) : 1.0f;

        glFinish();
    }

    inline void SumLayerGradsFast(const GLuint& target_index, const GLuint& layer_index, const GLuint& layer_size, const GLuint& workgroup_size)
    {
        BindShader(fSumGradsProg);
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, weightSSBOs[layer_index]);
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, layerSSBOs[layer_index]);
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, layerSSBOs[layer_index+1]);

        glUniform1ui(wswid, layer_size);
        glUniform1ui(wsoid, target_index);

        assert(layer_size % workgroup_size == 0);
        glDispatchCompute(layer_size / workgroup_size, 1, 1);
        glMemoryBarrier(GL_ALL_BARRIER_BITS);
        glFinish();
    }

    inline void ForwardProp(MemNet& net, const GLuint& workgroup_size)
    {
        const std::vector<GLuint>& netShape(net.Shape());

        BindShader(forwardInProg);
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, inputSSBO);
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, layerSSBOs[0]);

        assert(netShape[0] % workgroup_size == 0);
        glDispatchCompute(netShape[0] / workgroup_size, 1, 1);
        glMemoryBarrier(GL_ALL_BARRIER_BITS);
        glFinish();

        BindShader(forwardNProg);

        GLuint prevSize = netShape[0];

        for (GLuint i=1; i < netShape.size()-1; ++i)
        {
            glUniform1ui(fnwid, prevSize);
            glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, weightSSBOs[i-1]);
            glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, layerSSBOs[i]);
            glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, layerSSBOs[i-1]);

            assert(netShape[i] % workgroup_size == 0);
            glDispatchCompute(netShape[i] / workgroup_size, 1, 1);
            glMemoryBarrier(GL_ALL_BARRIER_BITS);
            glFinish();

            prevSize = netShape[i];
        }

        BindShader(forwardOutProg);

        glUniform1ui(fowid, prevSize);
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, weightSSBOs[netShape.size()-2]);
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, layerSSBOs[netShape.size()-1]);
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, layerSSBOs[netShape.size()-2]);

        assert(net.OutputSize() % workgroup_size == 0);
        glDispatchCompute(net.OutputSize() / workgroup_size, 1, 1);
        glMemoryBarrier(GL_ALL_BARRIER_BITS);
        glFinish();
    }

    inline void BackwardProp(MemNet& net, const GLuint& workgroup_size)
    {
        const std::vector<GLuint>& netShape(net.Shape());
        GLuint outputSize = netShape[netShape.size()-1];
        GLuint prevSize = netShape[netShape.size()-2];

        BindShader(backwardOutProg);
        glUniform1ui(bowid, prevSize);

        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, outputSSBO);
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, errorSSBO);
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, gradSSBOs[netShape.size()-2]);
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 3, layerSSBOs[netShape.size()-1]);
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 4, layerSSBOs[netShape.size()-2]);

        assert(outputSize % workgroup_size == 0);
        glDispatchCompute(outputSize / workgroup_size, 1, 1);
        glMemoryBarrier(GL_ALL_BARRIER_BITS);
        glFinish();

        SumLayerGrads(netShape.size()-2, prevSize, outputSize, workgroup_size);

        for (GLuint i=netShape.size()-2; i>0; --i)
        {
            BindShader(backwardNProg);
            prevSize = netShape[i-1];
            glUniform1ui(bnwid, prevSize);

            glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, gradSSBOs[i-1]);
            glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, layerSSBOs[i]);
            glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, layerSSBOs[i-1]);

            assert(netShape[i] % workgroup_size == 0);
            glDispatchCompute(netShape[i] / workgroup_size, 1, 1);
            glMemoryBarrier(GL_ALL_BARRIER_BITS);
            glFinish();

            SumLayerGrads(i-1, prevSize, netShape[i], workgroup_size);
        }

        BindShader(backwardInProg);
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, layerSSBOs[0]);

        assert(netShape[0] % workgroup_size == 0);
        glDispatchCompute(netShape[0] / workgroup_size, 1, 1);
        glMemoryBarrier(GL_ALL_BARRIER_BITS);
        glFinish();
    }

    inline void BackwardPropFast(MemNet& net, std::vector<GLfloat>& errors,
                                const GLuint& target_index, const GLuint& workgroup_size)
    {
        const std::vector<GLuint>& netShape(net.Shape());
        GLuint prevSize = netShape[netShape.size()-2];

        CompTargNeuronsW2V(errors, netShape.size()-1, target_index);

        SumLayerGradsFast(target_index, netShape.size()-2, prevSize, workgroup_size);

        BindShader(fBackS1Prog);
        glUniform1ui(wbwid, prevSize);
        glUniform1ui(wboid, target_index);

        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, gradSSBOs[netShape.size()-2]);
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, layerSSBOs[netShape.size()-2]);
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, layerSSBOs[netShape.size()-1]);

        assert(prevSize % workgroup_size == 0);
        glDispatchCompute(prevSize / workgroup_size, 1, 1);
        glMemoryBarrier(GL_ALL_BARRIER_BITS);
        glFinish();

        for (GLuint i=netShape.size()-2; i>0; --i)
        {
            BindShader(backwardNProg);
            prevSize = netShape[i-1];
            glUniform1ui(bnwid, prevSize);

            glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, gradSSBOs[i-1]);
            glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, layerSSBOs[i]);
            glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, layerSSBOs[i-1]);

            assert(netShape[i] % workgroup_size == 0);
            glDispatchCompute(netShape[i] / workgroup_size, 1, 1);
            glMemoryBarrier(GL_ALL_BARRIER_BITS);
            glFinish();

            SumLayerGrads(i-1, prevSize, netShape[i], workgroup_size);
        }

        BindShader(backwardInProg);
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, layerSSBOs[0]);

        assert(netShape[0] % workgroup_size == 0);
        glDispatchCompute(netShape[0] / workgroup_size, 1, 1);
        glMemoryBarrier(GL_ALL_BARRIER_BITS);
        glFinish();
    }

    inline void SumLayerGrads(const GLuint& layer_index, const GLuint& layer_size,
                              const GLuint& next_layer_size, const GLuint& workgroup_size)
    {
        BindShader(sumGradsProg);
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, weightSSBOs[layer_index]);
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, layerSSBOs[layer_index]);
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, layerSSBOs[layer_index+1]);

        glUniform1ui(sgwid, layer_size);
        glUniform1ui(sgoid, next_layer_size);

        assert(layer_size % workgroup_size == 0);
        glDispatchCompute(layer_size / workgroup_size, 1, 1);
        glMemoryBarrier(GL_ALL_BARRIER_BITS);
        glFinish();
    }

    inline void UpdateErrors(MemNet& net, const GLuint& workgroup_size)
    {
        BindShader(updateErrorsProg);
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, outputSSBO);
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, errorSSBO);
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, layerSSBOs[net.LayerCount()-1]);
        glDispatchCompute(net.OutputSize() / workgroup_size, 1, 1);
        glMemoryBarrier(GL_ALL_BARRIER_BITS);
        glFinish();
    }

    inline void UpdateWeights(MemNet& net, const GLuint& workgroup_size)
    {
        BindShader(updateWeightsProg);

        for (GLuint i=1; i < net.Shape().size(); ++i)
        {
            glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, weightSSBOs[i-1]);
            glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, gradSSBOs[i-1]);
            assert(net.WeightCount(i) % workgroup_size == 0);
            glDispatchCompute(net.WeightCount(i) / workgroup_size, 1, 1);
            glMemoryBarrier(GL_ALL_BARRIER_BITS);
            glFinish();
        }
    }

    inline void UpdateNeurons(MemNet& net, const GLuint& workgroup_size)
    {
        const std::vector<GLuint>& netShape(net.Shape());
        BindShader(updateNeuronProg);

        for (GLuint i=0; i < netShape.size(); ++i)
        {
            glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, layerSSBOs[i]);
            glDispatchCompute(netShape[i] / workgroup_size, 1, 1);
            glMemoryBarrier(GL_ALL_BARRIER_BITS);
            glFinish();
        }
    }

    inline void ResetNetMem(MemNet& net, const GLuint& workgroup_size)
    {
        const std::vector<GLuint>& netShape(net.Shape());
        BindShader(resetMemProg);

        for (GLuint i=0; i < netShape.size(); ++i)
        {
            glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, layerSSBOs[i]);
            glDispatchCompute(netShape[i] / workgroup_size, 1, 1);
            glMemoryBarrier(GL_ALL_BARRIER_BITS);
            glFinish();
        }
    }

    inline void CheckProgramStatus(const ShaderSource& shader, GLint check_type)
    {
        GLint result;
        glGetProgramiv(shader.programID, check_type, &result);

        if (result == GL_FALSE) {
            GLint length;
            glGetProgramiv(shader.programID, GL_INFO_LOG_LENGTH, &length);
            std::string message((size_t)length, ' ');
            glGetProgramInfoLog(shader.programID, length, &length, &message[0]);
            std::stringstream ss;
            ss << std::endl << "[OpenGL Error] Failed to ";
            ss << (check_type == GL_LINK_STATUS ? "link" : "validate");
            ss << " shader: " << shader.shaderFile << std::endl << message;
            HandleFatalError(ss.str());
        }
    }

    inline void CheckCompileStatus(const ShaderSource& shader)
    {
        GLint result;
        glGetShaderiv(shader.shaderID, GL_COMPILE_STATUS, &result);

        if (result == GL_FALSE) {
            GLint length;
            glGetShaderiv(shader.shaderID, GL_INFO_LOG_LENGTH, &length);
            std::string message((size_t)length, ' ');
            glGetShaderInfoLog(shader.shaderID, length, &length, &message[0]);
            std::stringstream ss;
            ss << std::endl << "[OpenGL Error] Failed to compile compute shader: ";
            ss << shader.shaderFile << std::endl << message;
            HandleFatalError(ss.str());
        }
    }

    inline std::string ErrorCodeToStr(const GLenum error_code)
    {
        switch (error_code) {
            case GL_INVALID_ENUM: return "invalid enumeration"; break;
            case GL_INVALID_VALUE: return "invalid value"; break;
            case GL_INVALID_OPERATION: return "invalid operation"; break;
            case GL_STACK_OVERFLOW: return "stack overflow"; break;
            case GL_STACK_UNDERFLOW: return "stack underflow"; break;
            case GL_OUT_OF_MEMORY: return "out of memory"; break;
            case GL_INVALID_FRAMEBUFFER_OPERATION: return "invalid framebuffer operation"; break;
            case GL_CONTEXT_LOST: return "context lost"; break;
            default: return "unknown error code"; break;
        }
    }

    inline bool PrintErrors()
    {
        bool errorFound = false;
        while (GLenum error = glGetError()) {
            std::cout << "\n[OpenGL Error] (" << std::to_string(error) << "): " << ErrorCodeToStr(error) << std::endl;
            errorFound = true;
        }
        return errorFound;
    }

};
