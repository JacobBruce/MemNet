#version 460 core
layout (local_size_x = WORKGROUP_SIZE, local_size_y = 1, local_size_z = 1) in;

NEURON_STRUCT;

layout(std430, binding = 0) buffer LayerBuffer
{
    Neuron neurons[];
};

void main()
{
	neurons[gl_GlobalInvocationID.x].bias -= LEARN_RATE_BIAS * neurons[gl_GlobalInvocationID.x].bgrad * GRAD_DIV;
	neurons[gl_GlobalInvocationID.x].frate -= LEARN_RATE_MEM * neurons[gl_GlobalInvocationID.x].bgrad * GRAD_DIV;
	neurons[gl_GlobalInvocationID.x].mweight -= LEARN_RATE * neurons[gl_GlobalInvocationID.x].mgrad * GRAD_DIV;
	
	neurons[gl_GlobalInvocationID.x].bgrad = 0.0;
	neurons[gl_GlobalInvocationID.x].mgrad = 0.0;
}