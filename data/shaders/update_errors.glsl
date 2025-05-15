#version 460 core
layout (local_size_x = WORKGROUP_SIZE, local_size_y = 1, local_size_z = 1) in;

#define POS_WEIGHT POS_WEIGHT_VAL
#define NEG_WEIGHT NEG_WEIGHT_VAL
#define ALPHA ALPHA_VAL
#define GAMMA GAMMA_VAL

float temp;

NEURON_STRUCT;

layout(std430, binding = 0) buffer TargetBuffer
{
    float targets[];
};

layout(std430, binding = 1) buffer ErrorBuffer
{
    float errors[];
};

layout(std430, binding = 2) buffer LayerBuffer
{
    Neuron neurons[];
};

float CalcError(float a, float b)
{
	LOSS_FUNC;
}

void main()
{
	errors[gl_GlobalInvocationID.x] += 
		CalcError(neurons[gl_GlobalInvocationID.x].actout, targets[gl_GlobalInvocationID.x]);
}