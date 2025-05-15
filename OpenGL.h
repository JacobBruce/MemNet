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

        ReplaceStr(shaderSource, "NEURON_STRUCT", glslNeuronStr);
        ReplaceStr(shaderSource, "GRAD_DIV", std::to_string(gradDiv));
        ReplaceStr(shaderSource, "WORKGROUP_SIZE", std::to_string(GLOBALS::WORKGROUP_SIZE));
        ReplaceStr(shaderSource, "LEARN_RATE_BIAS", GLOBALS::config_map["LEARN_RATE_BIAS"]);
        ReplaceStr(shaderSource, "LEARN_RATE_MEM", GLOBALS::config_map["LEARN_RATE_MEM"]);
        ReplaceStr(shaderSource, "LEARN_RATE", GLOBALS::config_map["LEARN_RATE"]);
        ReplaceStr(shaderSource, "MOMENTUM", GLOBALS::config_map["MOMENTUM"]);
        ReplaceStr(shaderSource, "TRAIN_STEPS", GLOBALS::config_map["TRAIN_STEPS"]);
        ReplaceStr(shaderSource, "MEM_BLEND_FUNC", GLOBALS::config_map["MEM_BLEND_FUNC"]);
        ReplaceStr(shaderSource, "ACT_DERIV_OUT", GLOBALS::config_map["ACT_D_OUT"]);
        ReplaceStr(shaderSource, "ACT_FUNC", GLOBALS::config_map["ACT_F"]);
        ReplaceStr(shaderSource, "ACT_DERIV", GLOBALS::config_map["ACT_D"]);
        ReplaceStr(shaderSource, "LOSS_FUNC", GLOBALS::config_map["LOSS_F"]);
        ReplaceStr(shaderSource, "LOSS_DERIV", GLOBALS::config_map["LOSS_D"]);
        ReplaceStr(shaderSource, "POS_WEIGHT_VAL", GLOBALS::config_map["POS_WEIGHT"]);
        ReplaceStr(shaderSource, "NEG_WEIGHT_VAL", GLOBALS::config_map["NEG_WEIGHT"]);
        ReplaceStr(shaderSource, "ALPHA_VAL", GLOBALS::config_map["FL_ALPHA"]);
        ReplaceStr(shaderSource, "GAMMA_VAL", GLOBALS::config_map["FL_GAMMA"]);

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
    std::vector<ShaderSource> shaders;
    std::vector<ShaderSource> nShaders;

    std::vector<GLuint> layerSSBOs;
    std::vector<GLuint> weightSSBOs;
    std::vector<GLuint> gradSSBOs;

    GLuint errorSSBO;
    GLuint inputSSBO;
    GLuint targetSSBO;

    std::vector<GLuint> forwardNProgs;
    std::vector<GLuint> forwardNShaders;
    std::vector<GLuint> forwardNNMProgs;
    std::vector<GLuint> forwardNNMShaders;
    std::vector<GLuint> backwardNProgs;
    std::vector<GLuint> backwardNShaders;
    std::vector<GLuint> backwardNNMProgs;
    std::vector<GLuint> backwardNNMShaders;
    GLuint forwardInProg;
    GLuint forwardInShader;
    GLuint backwardInProg;
    GLuint backwardInShader;
    GLuint backwardOutProg;
    GLuint backwardOutShader;
    GLuint softmaxS1Prog;
    GLuint softmaxS1Shader;
    GLuint softmaxS2Prog;
    GLuint softmaxS2Shader;
    GLuint ioToTargsProg;
    GLuint ioToTargsShader;
    GLuint updateWeightsProg;
    GLuint updateWeightsShader;
    GLuint updateErrorsProg;
    GLuint updateErrorsShader;
    GLuint updateNeuronProg;
    GLuint updateNeuronShader;
    GLuint updateNeuronNMShader;
    GLuint updateNeuronNMProg;
    GLuint resetMemProg;
    GLuint resetMemShader;
    GLuint sumGradsProg;
    GLuint sumGradsShader;
    GLuint activeShader;
    GLuint bowid, sgwid, sgoid,
           smnid, smsid, smoid;

    std::vector<GLuint> fnwids, bnwids,
                        fnwnds, bnwnds;

public:

    GL() {}

    ~GL()
    {
        glDeleteBuffers(layerSSBOs.size(), layerSSBOs.data());
        glDeleteBuffers(weightSSBOs.size(), weightSSBOs.data());
        glDeleteBuffers(gradSSBOs.size(), gradSSBOs.data());
        glDeleteBuffers(1, &errorSSBO);
        glDeleteBuffers(1, &inputSSBO);
        glDeleteBuffers(1, &targetSSBO);
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
        glGenBuffers(1, &errorSSBO);
        glGenBuffers(1, &inputSSBO);
        glGenBuffers(1, &targetSSBO);
    }

    void SetupShaders(const std::vector<std::string>& act_funcs, const bool use_avg_shaders)
    {
        if (act_funcs.size() < 2) HandleFatalError("Net must have at least two layers!");

        MemNet::ApplyActFunc(act_funcs[0]);
        shaders.emplace_back(GLOBALS::DATA_FOLDER+"shaders/forward_in.glsl", forwardInShader, forwardInProg);
        shaders.emplace_back(GLOBALS::DATA_FOLDER+"shaders/softmax_s1.glsl", softmaxS1Shader, softmaxS1Prog);
        shaders.emplace_back(GLOBALS::DATA_FOLDER+"shaders/softmax_s2.glsl", softmaxS2Shader, softmaxS2Prog);
        shaders.emplace_back(GLOBALS::DATA_FOLDER+"shaders/in_outs_to_targs.glsl", ioToTargsShader, ioToTargsProg);

        if (use_avg_shaders) {
            shaders.emplace_back(GLOBALS::DATA_FOLDER+"shaders/backward_in_avg.glsl", backwardInShader, backwardInProg);
            MemNet::ApplyActFunc(act_funcs.back());
            shaders.emplace_back(GLOBALS::DATA_FOLDER+"shaders/backward_out_avg.glsl", backwardOutShader, backwardOutProg);
            shaders.emplace_back(GLOBALS::DATA_FOLDER+"shaders/update_weights_avg.glsl", updateWeightsShader, updateWeightsProg);
            shaders.emplace_back(GLOBALS::DATA_FOLDER+"shaders/update_neuron.glsl", updateNeuronShader, updateNeuronProg);
            shaders.emplace_back(GLOBALS::DATA_FOLDER+"shaders/nomem/update_neuron.glsl", updateNeuronNMShader, updateNeuronNMProg);
        } else {
            shaders.emplace_back(GLOBALS::DATA_FOLDER+"shaders/backward_in.glsl", backwardInShader, backwardInProg);
            MemNet::ApplyActFunc(act_funcs.back());
            shaders.emplace_back(GLOBALS::DATA_FOLDER+"shaders/backward_out.glsl", backwardOutShader, backwardOutProg);
            shaders.emplace_back(GLOBALS::DATA_FOLDER+"shaders/update_weights.glsl", updateWeightsShader, updateWeightsProg);
        }

        shaders.emplace_back(GLOBALS::DATA_FOLDER+"shaders/sum_ngrads.glsl", sumGradsShader, sumGradsProg);
        shaders.emplace_back(GLOBALS::DATA_FOLDER+"shaders/update_errors.glsl", updateErrorsShader, updateErrorsProg);
        shaders.emplace_back(GLOBALS::DATA_FOLDER+"shaders/reset_mem.glsl", resetMemShader, resetMemProg);

        forwardNShaders.resize(act_funcs.size()-1);
        forwardNProgs.resize(act_funcs.size()-1);
        backwardNShaders.resize(act_funcs.size()-2);
        backwardNProgs.resize(act_funcs.size()-2);

        forwardNNMShaders.resize(act_funcs.size()-1);
        forwardNNMProgs.resize(act_funcs.size()-1);
        backwardNNMShaders.resize(act_funcs.size()-2);
        backwardNNMProgs.resize(act_funcs.size()-2);

        for (GLuint i=0; i < act_funcs.size()-1; ++i)
        {
            MemNet::ApplyActFunc(act_funcs[i+1]);
            nShaders.emplace_back(GLOBALS::DATA_FOLDER+"shaders/forward_n.glsl", forwardNShaders[i], forwardNProgs[i]);
            nShaders.emplace_back(GLOBALS::DATA_FOLDER+"shaders/nomem/forward_n.glsl", forwardNNMShaders[i], forwardNNMProgs[i]);
        }

        for (GLuint i=0; i < act_funcs.size()-2; ++i)
        {
            MemNet::ApplyActFunc(act_funcs[i+1]);

            if (use_avg_shaders) {
                nShaders.emplace_back(GLOBALS::DATA_FOLDER+"shaders/backward_n_avg.glsl", backwardNShaders[i], backwardNProgs[i]);
                nShaders.emplace_back(GLOBALS::DATA_FOLDER+"shaders/nomem/backward_n_avg.glsl", backwardNNMShaders[i], backwardNNMProgs[i]);
            } else {
                nShaders.emplace_back(GLOBALS::DATA_FOLDER+"shaders/backward_n.glsl", backwardNShaders[i], backwardNProgs[i]);
                nShaders.emplace_back(GLOBALS::DATA_FOLDER+"shaders/nomem/backward_n.glsl", backwardNNMShaders[i], backwardNNMProgs[i]);
            }
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

        for (GLuint i=0; i < nShaders.size(); ++i)
        {
            // Compile shader from source
            nShaders[i].shaderID = glCreateShader(GL_COMPUTE_SHADER);
            glShaderSource(nShaders[i].shaderID, 1, nShaders[i].GetGLPtr(), nullptr);
            glCompileShader(nShaders[i].shaderID);
            CheckCompileStatus(nShaders[i]);

            // Attach and link shader to shader program
            nShaders[i].programID = glCreateProgram();
            glAttachShader(nShaders[i].programID, nShaders[i].shaderID);
            glLinkProgram(nShaders[i].programID);

            // Check shader program is valid
            CheckProgramStatus(nShaders[i], GL_LINK_STATUS);
            glValidateProgram(nShaders[i].programID);
            CheckProgramStatus(nShaders[i], GL_VALIDATE_STATUS);
        }

        // Setup compute shader uniforms

        fnwids.resize(forwardNProgs.size());
        fnwnds.resize(forwardNNMProgs.size());
        bnwids.resize(backwardNProgs.size());
        bnwnds.resize(backwardNNMProgs.size());

        for (GLuint i=0; i < fnwids.size(); ++i)
        {
            glUseProgram(forwardNProgs[i]);
            fnwids[i] = glGetUniformLocation(forwardNProgs[i], "weightsPerNeuron");

            glUseProgram(forwardNNMProgs[i]);
            fnwnds[i] = glGetUniformLocation(forwardNNMProgs[i], "weightsPerNeuron");
        }

        for (GLuint i=0; i < bnwids.size(); ++i)
        {
            glUseProgram(backwardNProgs[i]);
            bnwids[i] = glGetUniformLocation(backwardNProgs[i], "weightsPerNeuron");

            glUseProgram(backwardNNMProgs[i]);
            bnwnds[i] = glGetUniformLocation(backwardNNMProgs[i], "weightsPerNeuron");
        }

        glUseProgram(backwardOutProg);
        bowid = glGetUniformLocation(backwardOutProg, "weightsPerNeuron");

        glUseProgram(sumGradsProg);
        sgwid = glGetUniformLocation(sumGradsProg, "weightsPerNeuron");
        sgoid = glGetUniformLocation(sumGradsProg, "otherLayerSize");

        glUseProgram(softmaxS1Prog);
        smnid = glGetUniformLocation(softmaxS1Prog, "weightsPerNeuron");

        glUseProgram(softmaxS2Prog);
        smsid = glGetUniformLocation(softmaxS2Prog, "softmaxDivisor");
        smoid = glGetUniformLocation(softmaxS2Prog, "outputOffset");
    }

    void Initialize(const std::vector<std::string>& act_funcs, bool use_avg_shaders)
    {
        // Setup buffers and shaders
        SetupMainBuffers(act_funcs.size());
        SetupShaders(act_funcs, use_avg_shaders);

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

    inline void AllocErrorSSBO(GLuint out_layer_size, const void* zero_inputs)
    {
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, errorSSBO);
        glBufferData(GL_SHADER_STORAGE_BUFFER, sizeof(GLfloat)*out_layer_size, zero_inputs, GL_DYNAMIC_DRAW);
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
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, targetSSBO);
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

    inline void CopyIndexOutToSSBO(GLuint& target_index, GLuint& last_out_index)
    {
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, targetSSBO);
        glBufferSubData(GL_SHADER_STORAGE_BUFFER, sizeof(GLfloat)*last_out_index, sizeof(GLfloat), &GLOBALS::NEG_LABEL);
        glBufferSubData(GL_SHADER_STORAGE_BUFFER, sizeof(GLfloat)*target_index, sizeof(GLfloat), &GLOBALS::POS_LABEL);
        glFinish();
    }

    inline void CopyIndexOutsToSSBO(GLuint& target1_index, GLuint& target2_index, GLuint& last_out1_index, GLuint& last_out2_index)
    {
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, targetSSBO);
        glBufferSubData(GL_SHADER_STORAGE_BUFFER, sizeof(GLfloat)*last_out1_index, sizeof(GLfloat), &GLOBALS::NEG_LABEL);
        glBufferSubData(GL_SHADER_STORAGE_BUFFER, sizeof(GLfloat)*last_out2_index, sizeof(GLfloat), &GLOBALS::NEG_LABEL);
        glBufferSubData(GL_SHADER_STORAGE_BUFFER, sizeof(GLfloat)*target1_index, sizeof(GLfloat), &GLOBALS::POS_LABEL);
        glBufferSubData(GL_SHADER_STORAGE_BUFFER, sizeof(GLfloat)*target2_index, sizeof(GLfloat), &GLOBALS::POS_LABEL);
        glFinish();
    }

    /*inline void CopyInputToSSBO(const GLuint& input_index, const GLfloat& val)
    {
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, inputSSBO);
        glBufferSubData(GL_SHADER_STORAGE_BUFFER, sizeof(GLfloat)*input_index, sizeof(GLfloat), &val);
        glFinish();
    }

    inline void CopyTargetToSSBO(const GLuint& target_index, const GLfloat& val)
    {
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, targetSSBO);
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
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, targetSSBO);
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

    /*inline void CopyOutputLayer(MemNet& net, std::vector<Neuron>& dest)
    {
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, layerSSBOs[net.LayerCount()-1]);

        glGetBufferSubData(
            GL_SHADER_STORAGE_BUFFER, 0,
            sizeof(Neuron)*dest.size(), dest.data()
        );

        glFinish();
    }

    inline void CompTargErrorFast(std::vector<GLfloat>& errors, const GLuint& layer_index, const GLuint& target_index)
    {
        Neuron targNeuron;

        glBindBuffer(GL_SHADER_STORAGE_BUFFER, layerSSBOs[layer_index]);

        glGetBufferSubData(
            GL_SHADER_STORAGE_BUFFER, target_index*sizeof(Neuron),
            sizeof(Neuron), &targNeuron
        );

        glFinish();

        targNeuron.grad = (1.f - (targNeuron.actout * targNeuron.actout)) * (-1.f / targNeuron.actout);
        //targNeuron.bgrad += outGrad;
        //targNeuron.mgrad += outGrad * targNeuron.mprev;

        glBufferSubData(
            GL_SHADER_STORAGE_BUFFER, target_index*sizeof(Neuron),
            sizeof(Neuron), &targNeuron
        );

        errors[target_index] += -std::log(std::min(std::max(targNeuron.actout, 0.00000001f), 0.99999999f));

        glFinish();
    }

    inline void SumLayerGradsFast(const GLuint& target_index, const GLuint& layer_index,
                                  const GLuint& layer_size, bool& first_run, const GLuint& workgroup_size)
    {
        if (first_run) {
            first_run = false;
            BindShader(fSumGradsProg);
            glUniform1ui(wswid, layer_size);
            glUniform1ui(wsoid, target_index);
        } else {
            BindShader(fSumGradsAddProg);
            glUniform1ui(wawid, layer_size);
            glUniform1ui(waoid, target_index);
        }

        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, weightSSBOs[layer_index]);
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, layerSSBOs[layer_index]);
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, layerSSBOs[layer_index+1]);

        assert(layer_size % workgroup_size == 0);
        glDispatchCompute(layer_size / workgroup_size, 1, 1);
        glMemoryBarrier(GL_ALL_BARRIER_BITS);
        glFinish();
    }*/

    inline void CopyInOutsToTargs(const GLuint& layer_size, const GLuint& workgroup_size)
    {
        BindShader(ioToTargsProg);
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, targetSSBO);
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, layerSSBOs[0]);

        assert(layer_size % workgroup_size == 0);
        glDispatchCompute(layer_size / workgroup_size, 1, 1);
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

    inline void ComputeLayerN(const std::vector<GLuint>& net_shape, const GLuint& layer_index, const GLuint& workgroup_size)
    {
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, weightSSBOs[layer_index-1]);
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, layerSSBOs[layer_index]);
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, layerSSBOs[layer_index-1]);

        assert(net_shape[layer_index] % workgroup_size == 0);
        glDispatchCompute(net_shape[layer_index] / workgroup_size, 1, 1);
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

        GLuint prevSize = netShape[0];

        for (GLuint i=1; i < netShape.size(); ++i)
        {
            if (net.LayerUsesSoftMax(i)) {

                BindShader(softmaxS1Prog);
                glUniform1ui(smnid, prevSize);
                ComputeLayerN(netShape, i, workgroup_size);

                CopyNeuronsFromSSBO(net, i);
                Neuron* neurons(net.LayerNeurons(i));

                BindShader(softmaxS2Prog);
                glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, layerSSBOs[i]);

                assert(net.OutputSpan() % workgroup_size == 0);
                GLuint workGroups = net.OutputSpan() / workgroup_size;

                for (GLuint r=0; r < net.OutputRows(); ++r)
                {
                    GLfloat softmaxSum = 0.f;
                    GLuint outOffset = r * net.OutputSpan();

                    for (GLuint n=0; n < net.OutputSpan(); ++n)
                        softmaxSum += neurons[outOffset+n].temp;

                    if (softmaxSum == 0.f) softmaxSum = FLOAT_EPSILON;

                    glUniform1f(smsid, softmaxSum);
                    glUniform1ui(smoid, outOffset);

                    glDispatchCompute(workGroups, 1, 1);
                }

                glMemoryBarrier(GL_ALL_BARRIER_BITS);
                glFinish();

            } else if (net.LayerHasMem(i)) {

                BindShader(forwardNProgs[i-1]);
                glUniform1ui(fnwids[i-1], prevSize);
                ComputeLayerN(netShape, i, workgroup_size);

            } else {

                BindShader(forwardNNMProgs[i-1]);
                glUniform1ui(fnwnds[i-1], prevSize);
                ComputeLayerN(netShape, i, workgroup_size);
            }

            prevSize = netShape[i];
        }
    }

    inline void BackwardProp(MemNet& net, const GLuint& workgroup_size)
    {
        const std::vector<GLuint>& netShape(net.Shape());
        GLuint outputSize = netShape[netShape.size()-1];
        GLuint prevSize = netShape[netShape.size()-2];

        BindShader(backwardOutProg);
        glUniform1ui(bowid, prevSize);

        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, targetSSBO);
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, errorSSBO);
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, gradSSBOs[netShape.size()-2]);
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 3, layerSSBOs[netShape.size()-1]);
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 4, layerSSBOs[netShape.size()-2]);

        assert(outputSize % workgroup_size == 0);
        glDispatchCompute(outputSize / workgroup_size, 1, 1);
        glMemoryBarrier(GL_ALL_BARRIER_BITS);
        glFinish();

        SumLayerGrads(netShape.size()-2, prevSize, outputSize, workgroup_size);

        for (GLuint i=netShape.size()-2; i>GLOBALS::BP_STOP; --i)
        {
            prevSize = netShape[i-1];

            if (net.LayerHasMem(i)) {
                BindShader(backwardNProgs[i-1]);
                glUniform1ui(bnwids[i-1], prevSize);
            } else {
                BindShader(backwardNNMProgs[i-1]);
                glUniform1ui(bnwnds[i-1], prevSize);
            }

            glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, gradSSBOs[i-1]);
            glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, layerSSBOs[i]);
            glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, layerSSBOs[i-1]);

            assert(netShape[i] % workgroup_size == 0);
            glDispatchCompute(netShape[i] / workgroup_size, 1, 1);
            glMemoryBarrier(GL_ALL_BARRIER_BITS);
            glFinish();

            SumLayerGrads(i-1, prevSize, netShape[i], workgroup_size);
        }

        if (net.LayerCount() == GLOBALS::BP_DEPTH) {
            BindShader(backwardInProg);
            glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, layerSSBOs[0]);

            assert(netShape[0] % workgroup_size == 0);
            glDispatchCompute(netShape[0] / workgroup_size, 1, 1);
            glMemoryBarrier(GL_ALL_BARRIER_BITS);
            glFinish();
        }
    }

    /*inline void BackwardPropFast(MemNet& net, std::vector<GLfloat>& errors,
                                const std::vector<GLuint>& target_indices, const GLuint& workgroup_size)
    {
        const std::vector<GLuint>& netShape(net.Shape());
        GLuint prevSize = netShape[netShape.size()-2];
        bool firstGradRun = true;

        for (const GLuint& target_index : target_indices)
        {
            CompTargErrorFast(errors, netShape.size()-1, target_index);
            SumLayerGradsFast(target_index, netShape.size()-2, prevSize, firstGradRun, workgroup_size);
        }

        BindShader(fBackS1Prog);
        glUniform1ui(wbwid, prevSize);

        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, gradSSBOs[netShape.size()-2]);
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, layerSSBOs[netShape.size()-2]);
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, layerSSBOs[netShape.size()-1]);

        for (const GLuint& target_index : target_indices)
        {
            glUniform1ui(wboid, target_index);

            assert(prevSize % workgroup_size == 0);
            glDispatchCompute(prevSize / workgroup_size, 1, 1);
        }

        glMemoryBarrier(GL_ALL_BARRIER_BITS);
        glFinish();

        for (GLuint i=netShape.size()-2; i>0; --i)
        {
            prevSize = netShape[i-1];

            if (net.LayerHasMem(i)) {
                BindShader(backwardNProgs[i-1]);
                glUniform1ui(bnwids[i-1], prevSize);
            } else {
                BindShader(backwardNNMProgs[i-1]);
                glUniform1ui(bnwnds[i-1], prevSize);
            }

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
    }*/

    inline void UpdateErrors(MemNet& net, const GLuint& workgroup_size)
    {
        BindShader(updateErrorsProg);
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, targetSSBO);
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, errorSSBO);
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, layerSSBOs[net.LayerCount()-1]);
        glDispatchCompute(net.OutputSize() / workgroup_size, 1, 1);
        glMemoryBarrier(GL_ALL_BARRIER_BITS);
        glFinish();
    }

    inline void UpdateWeights(MemNet& net, const GLuint& workgroup_size)
    {
        BindShader(updateWeightsProg);

        for (GLuint i=net.LayerCount()-1; i > GLOBALS::BP_STOP; --i)
        {
            glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, weightSSBOs[i-1]);
            glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, gradSSBOs[i-1]);
            assert(net.WeightCount(i) % workgroup_size == 0);
            glDispatchCompute(net.WeightCount(i) / workgroup_size, 1, 1);
        }

        glMemoryBarrier(GL_ALL_BARRIER_BITS);
        glFinish();
    }

    inline void UpdateNeurons(MemNet& net, const GLuint& workgroup_size)
    {
        const std::vector<GLuint>& netShape(net.Shape());

        for (GLuint i=net.LayerCount()-GLOBALS::BP_DEPTH; i < net.LayerCount(); ++i)
        {
            if (net.LayerHasMem(i)) {
                BindShader(updateNeuronProg);
            } else {
                BindShader(updateNeuronNMProg);
            }

            glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, layerSSBOs[i]);
            glDispatchCompute(netShape[i] / workgroup_size, 1, 1);
        }

        glMemoryBarrier(GL_ALL_BARRIER_BITS);
        glFinish();
    }

    inline void ResetNetMem(MemNet& net, const GLuint& workgroup_size)
    {
        const std::vector<GLuint>& netShape(net.Shape());
        BindShader(resetMemProg);

        for (GLuint i=0; i < netShape.size(); ++i)
        {
            if (net.LayerHasMem(i)) {
                glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, layerSSBOs[i]);
                glDispatchCompute(netShape[i] / workgroup_size, 1, 1);
            }
        }

        glMemoryBarrier(GL_ALL_BARRIER_BITS);
        glFinish();
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
