#version 460 core
layout (local_size_x = WORKGROUP_SIZE, local_size_y = 1, local_size_z = 1) in;

layout(std430, binding = 0) buffer WeightBuffer
{
    float weights[];
};

layout(std430, binding = 1) buffer GradientBuffer
{
    float gradients[];
};

void main()
{
	weights[gl_GlobalInvocationID.x] -= LEARN_RATE * gradients[gl_GlobalInvocationID.x];
}