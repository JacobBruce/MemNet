#version 460 core
layout (local_size_x = WORKGROUP_SIZE, local_size_y = 1, local_size_z = 1) in;

uniform float softmaxDivisor;
uniform uint outputOffset;

NEURON_STRUCT;

layout(std430, binding = 0) buffer LayerBuffer
{
    Neuron neurons[];
};

float BlendMem(float a, float b)
{
	return MEM_BLEND_FUNC;
}

void main()
{
	uint neuronIndex = gl_GlobalInvocationID.x + outputOffset;
	float memVal = neurons[neuronIndex].mem;
	float outVal = neurons[neuronIndex].temp / softmaxDivisor;
	
	neurons[neuronIndex].mem = BlendMem(outVal, neurons[neuronIndex].frate * memVal);
	neurons[neuronIndex].mprev = memVal;
	neurons[neuronIndex].actout = outVal;
}