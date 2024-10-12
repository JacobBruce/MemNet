#version 460 core
layout (local_size_x = WORKGROUP_SIZE, local_size_y = 1, local_size_z = 1) in;

NEURON_STRUCT;

layout(std430, binding = 0) buffer InputBuffer
{
    float inputs[];
};

layout(std430, binding = 1) buffer LayerBuffer
{
    Neuron neurons[];
};

float BlendMem(float a, float b)
{
	return MEM_BLEND_FUNC;
}

float ActFunc(float x)
{
	neurons[gl_GlobalInvocationID.x].insum = x;
	IN_ACT_FUNC;
}

void main()
{
	float memVal = neurons[gl_GlobalInvocationID.x].mem;
	
	float outVal = inputs[gl_GlobalInvocationID.x] + neurons[gl_GlobalInvocationID.x].bias +
				  (neurons[gl_GlobalInvocationID.x].mem * neurons[gl_GlobalInvocationID.x].mweight);
	
	outVal = ActFunc(outVal);
	
	neurons[gl_GlobalInvocationID.x].mem = BlendMem(outVal, neurons[gl_GlobalInvocationID.x].frate * memVal);
	neurons[gl_GlobalInvocationID.x].mprev = memVal;
	neurons[gl_GlobalInvocationID.x].actout = outVal;
}