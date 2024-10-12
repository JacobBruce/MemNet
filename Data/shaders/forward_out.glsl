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

float BlendMem(float a, float b)
{
	return MEM_BLEND_FUNC;
}

float ActFunc(float x)
{
	neurons[gl_GlobalInvocationID.x].insum = x;
	OUT_ACT_FUNC;
}

void main()
{
	uint weightIndex = gl_GlobalInvocationID.x * weightsPerNeuron;
	float memVal = neurons[gl_GlobalInvocationID.x].mem;
	float outVal = 0.0;
	
	for (uint i=0; i < weightsPerNeuron; ++i)
		outVal += weights[weightIndex+i] * otherNeurons[i].actout;
	
	outVal += neurons[gl_GlobalInvocationID.x].bias +
			  (memVal * neurons[gl_GlobalInvocationID.x].mweight);
		
	outVal = ActFunc(outVal);
	
	neurons[gl_GlobalInvocationID.x].mem = BlendMem(outVal, neurons[gl_GlobalInvocationID.x].frate * memVal);
	neurons[gl_GlobalInvocationID.x].mprev = memVal;
	neurons[gl_GlobalInvocationID.x].actout = outVal;
}