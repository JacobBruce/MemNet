#version 460 core
layout (local_size_x = WORKGROUP_SIZE, local_size_y = 1, local_size_z = 1) in;

uniform uint weightsPerNeuron;

NEURON_STRUCT;

layout(std430, binding = 0) buffer WeightBuffer
{
    float weights[];
};

layout(std430, binding = 1) buffer LayerBuffer
{
    Neuron neurons[];
};

layout(std430, binding = 2) buffer PrevLayerBuffer
{
    Neuron otherNeurons[];
};

float ActFunc(float x)
{
	neurons[gl_GlobalInvocationID.x].insum = x;
	ACT_FUNC;
}

void main()
{
	uint weightIndex = gl_GlobalInvocationID.x * weightsPerNeuron;
	float outVal = 0.0;
	
	for (uint i=0; i < weightsPerNeuron; ++i)
		outVal += weights[weightIndex+i] * otherNeurons[i].actout;
	
	outVal += neurons[gl_GlobalInvocationID.x].bias;
	
	neurons[gl_GlobalInvocationID.x].actout = ActFunc(outVal);
}