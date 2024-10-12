#pragma once
#include <cstdlib>
#include <string>
#include <iostream>
#include <fstream>
#include <algorithm>
#include <thread>
#include <atomic>
#include "Timer.h"
#include "OpenGL.h"
#include "MemNet.h"

class Engine
{
public:
    void Init();
    Engine(const bool use_train_steps);
	Engine(std::string net_dir, std::string net_name);
	Engine(const std::vector<GLuint>& net_shape, std::string net_type, GLfloat min_weight, GLfloat max_weight,
           std::string in_act_func, std::string hid_act_func, std::string out_act_func, std::string mem_blend_func);
	void GetNetOutputs(std::vector<GLfloat>& dest);
	void TestNet();
	void TrainNet(const GLuint train_epochs, const GLuint train_steps);
	void TrainTextNet(const GLuint train_epochs, const GLuint train_steps);
	void TrainWord2VecNet(const GLuint train_epochs, const GLuint train_steps);
	void ProcessData(const std::string& val, const std::string& func, const GLfloat& div_arg, std::vector<GLfloat>& dest, GLuint array_size);
	void LoadDataBatches(std::string data_file, std::string data_type, std::string input_func, std::string output_func, GLuint examples_per_batch);
	void CheckTextFiles(const std::string& data_file, float min_byte_token_ratio=2.0f);

	MemNet net;

private:
	GL openGL;

	bool useTrainSteps;

    std::vector<std::vector<GLfloat>> inBatches;
    std::vector<std::vector<GLfloat>> outBatches;

    std::vector<std::vector<uint32_t>> tokenBatches;
    phmap::parallel_flat_hash_map<std::u32string, uint32_t> wordMap;
};
