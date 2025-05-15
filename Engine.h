#pragma once
#include "DataLoad.h"
#include "Timer.h"
#include "OpenGL.h"
#include "MemNet.h"

class Engine
{
public:
    Engine(const bool use_train_steps);
	GL& GetGL();
    MemNet& GetNet();
	void SetNet(uint32_t net_index);
    void InitNet(MemNet& net);
    void LoadNet(std::string net_dir, std::string net_name);
	void SaveNet(std::string net_dir, std::string net_name);
    void GenNet(const std::vector<uint32_t>& net_shape, std::string net_type, float4 init_data,
            std::vector<std::string>& act_funcs, std::vector<std::string>& mem_layers, std::string mem_blend_func);
	void CopyLayerOutputs(MemNet& net, std::vector<GLfloat>& dest, GLuint layer_index);
	void GetNetOutputs(MemNet& net, std::vector<GLfloat>& dest);
	void RunNet();
	void TestNet();
	void TrainWord2Vec(const GLuint train_epochs, const GLuint train_steps);
	void TrainAutoEncoder(const GLuint train_epochs, const GLuint train_steps);
	void TrainTextAutoEncoder(const GLuint train_epochs, const GLuint train_steps);
	void TrainNet(const GLuint train_epochs, const GLuint train_steps);
	void TrainTextNet(const GLuint train_epochs, const GLuint train_steps);
    void LoadDataset(std::string data_file, std::string data_type, std::string input_func,
                         std::string output_func, uint32_t examples_per_batch);

private:
    uint32_t activeNetIndex;

	std::vector<MemNet> nets;
	std::vector<GL> glVec;

    std::vector<std::vector<GLfloat>> inBatches;
    std::vector<std::vector<GLfloat>> outBatches;

    std::vector<std::vector<uint32_t>> tokenBatches;
    phmap::parallel_flat_hash_map<std::u32string, uint32_t> wordMap;

	bool useTrainSteps;
};
