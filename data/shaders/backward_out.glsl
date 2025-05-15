#version 460 core
layout (local_size_x = WORKGROUP_SIZE, local_size_y = 1, local_size_z = 1) in;

#define POS_WEIGHT POS_WEIGHT_VAL
#define NEG_WEIGHT NEG_WEIGHT_VAL
#define ALPHA ALPHA_VAL
#define GAMMA GAMMA_VAL

uniform uint weightsPerNeuron;

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
	LOSS_FUNC;
}

float ErrorDeriv(float a, float b)
{
	LOSS_DERIV;
}

float ActDeriv(float y)
{
	ACT_DERIV_OUT;
}

void main()
{
	uint weightIndex = gl_GlobalInvocationID.x * weightsPerNeuron;
	float outVal = neurons[gl_GlobalInvocationID.x].actout;
	float target = targets[gl_GlobalInvocationID.x];
	
	errors[gl_GlobalInvocationID.x] += CalcError(outVal, target);
	
	float outGrad = ActDeriv(outVal) * ErrorDeriv(outVal, target);
	float biasGrad = outGrad + (neurons[gl_GlobalInvocationID.x].bgrad * MOMENTUM);
	float memGrad = (outGrad * neurons[gl_GlobalInvocationID.x].mprev) + (neurons[gl_GlobalInvocationID.x].mgrad * MOMENTUM);
	
	for (uint w=0; w < weightsPerNeuron; ++w)
	{
		gradients[weightIndex+w] = 
			(outGrad * otherNeurons[w].actout) + 
			(gradients[weightIndex+w] * MOMENTUM);
	}
	
	neurons[gl_GlobalInvocationID.x].grad = outGrad;
	neurons[gl_GlobalInvocationID.x].bgrad = biasGrad;
	neurons[gl_GlobalInvocationID.x].mgrad = memGrad;
	
	neurons[gl_GlobalInvocationID.x].bias -= LEARN_RATE_BIAS * biasGrad;
	neurons[gl_GlobalInvocationID.x].frate -= LEARN_RATE_MEM * biasGrad;
	neurons[gl_GlobalInvocationID.x].mweight -= LEARN_RATE * memGrad;
}