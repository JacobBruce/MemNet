#version 460 core
layout (local_size_x = WORKGROUP_SIZE, local_size_y = 1, local_size_z = 1) in;

uniform uint weightsPerNeuron;

NEURON_STRUCT;

layout(std430, binding = 0) buffer TargetBuffer
{
    float targets[];
};

layout(std430, binding = 1) buffer ErrorBuffer
{
    float errors[];
};

layout(std430, binding = 2) buffer GradientBuffer
{
    float gradients[];
};

layout(std430, binding = 3) buffer LayerBuffer
{
    Neuron neurons[];
};

layout(std430, binding = 4) buffer PrevLayerBuffer
{
    Neuron otherNeurons[];
};

float CalcError(float a, float b)
{
	return LOSS_FUNC;
}

float ErrorDeriv(float a, float b)
{
	return LOSS_DERIV;
}

float ActDeriv(float y)
{
	OUT_ACT_DERIV;
}

void main()
{
	uint weightIndex = gl_GlobalInvocationID.x * weightsPerNeuron;
	float outVal = neurons[gl_GlobalInvocationID.x].actout;
	float target = targets[gl_GlobalInvocationID.x];
	float outGrad = ActDeriv(outVal) * ErrorDeriv(target, outVal);
	
	neurons[gl_GlobalInvocationID.x].grad = outGrad;
	neurons[gl_GlobalInvocationID.x].bgrad += outGrad;
	neurons[gl_GlobalInvocationID.x].mgrad += outGrad * neurons[gl_GlobalInvocationID.x].mprev;
	
	for (uint w=0; w < weightsPerNeuron; ++w)
		gradients[weightIndex+w] += outGrad * otherNeurons[w].actout;
	
	errors[gl_GlobalInvocationID.x] += CalcError(target, outVal);
}