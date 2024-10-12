#version 460 core
layout (local_size_x = WORKGROUP_SIZE, local_size_y = 1, local_size_z = 1) in;

uniform uint weightsPerNeuron;
uniform uint outIndex;

NEURON_STRUCT;

layout(std430, binding = 0) buffer GradientBuffer
{
    float gradients[];
};

layout(std430, binding = 1) buffer LayerBuffer
{
    Neuron neurons[];
};

layout(std430, binding = 2) buffer OutLayerBuffer
{
    Neuron otherNeurons[];
};

void main()
{
	gradients[(outIndex * weightsPerNeuron) + gl_GlobalInvocationID.x] +=
		otherNeurons[outIndex].grad * neurons[gl_GlobalInvocationID.x].actout;
}