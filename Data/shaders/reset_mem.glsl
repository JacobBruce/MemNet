#version 460 core
layout (local_size_x = WORKGROUP_SIZE, local_size_y = 1, local_size_z = 1) in;

NEURON_STRUCT;

layout(std430, binding = 0) buffer LayerBuffer
{
    Neuron neurons[];
};

void main()
{
	neurons[gl_GlobalInvocationID.x].mem = 0.0;
	neurons[gl_GlobalInvocationID.x].mprev = 0.0;
}