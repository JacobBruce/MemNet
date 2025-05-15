#version 460 core
layout (local_size_x = WORKGROUP_SIZE, local_size_y = 1, local_size_z = 1) in;

NEURON_STRUCT;

layout(std430, binding = 0) buffer TargetBuffer
{
    float targets[];
};

layout(std430, binding = 1) buffer LayerBuffer
{
    Neuron neurons[];
};

void main()
{
	targets[gl_GlobalInvocationID.x] = neurons[gl_GlobalInvocationID.x].actout;
}