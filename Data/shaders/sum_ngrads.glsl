#version 460 core
layout (local_size_x = WORKGROUP_SIZE, local_size_y = 1, local_size_z = 1) in;

uniform uint weightsPerNeuron;
uniform uint otherLayerSize;

NEURON_STRUCT;

layout(std430, binding = 0) buffer WeightBuffer
{
    float weights[];
};

layout(std430, binding = 1) buffer LayerBuffer
{
    Neuron neurons[];
};

layout(std430, binding = 2) buffer NextLayerBuffer
{
    Neuron otherNeurons[];
};

void main()
{
	uint weightIndex = gl_GlobalInvocationID.x;
	float gradSum = 0.0;
	
	for (uint n=0; n < otherLayerSize; ++n)
	{
		gradSum += weights[weightIndex] * otherNeurons[n].grad;
		weightIndex += weightsPerNeuron;
	}
	
	neurons[gl_GlobalInvocationID.x].grad = gradSum;
}