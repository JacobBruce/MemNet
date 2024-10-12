#version 460 core
layout (local_size_x = WORKGROUP_SIZE, local_size_y = 1, local_size_z = 1) in;

uniform uint weightsPerNeuron;

NEURON_STRUCT;

layout(std430, binding = 0) buffer GradientBuffer
{
    float gradients[];
};

layout(std430, binding = 1) buffer LayerBuffer
{
    Neuron neurons[];
};

layout(std430, binding = 2) buffer PrevLayerBuffer
{
    Neuron otherNeurons[];
};

float ActDeriv(float y)
{
	HID_ACT_DERIV;
}

void main()
{
	uint weightIndex = gl_GlobalInvocationID.x * weightsPerNeuron;
	float outGrad = neurons[gl_GlobalInvocationID.x].grad;
	float biasGrad = outGrad + (neurons[gl_GlobalInvocationID.x].bgrad * MOMENTUM);
	float memGrad = (outGrad * neurons[gl_GlobalInvocationID.x].mprev) + (neurons[gl_GlobalInvocationID.x].mgrad * MOMENTUM);
	
	for (uint w=0; w < weightsPerNeuron; ++w)
	{
		gradients[weightIndex+w] = 
			(outGrad * otherNeurons[w].actout) + 
			(gradients[weightIndex+w] * MOMENTUM);
	}

	outGrad = (outGrad == 0.0) ? 
		ActDeriv(neurons[gl_GlobalInvocationID.x].actout) : 
		outGrad * ActDeriv(neurons[gl_GlobalInvocationID.x].actout);
		
	neurons[gl_GlobalInvocationID.x].grad = outGrad;
	neurons[gl_GlobalInvocationID.x].bgrad = biasGrad;
	neurons[gl_GlobalInvocationID.x].mgrad = memGrad;
	
	neurons[gl_GlobalInvocationID.x].bias -= LEARN_RATE_BIAS * biasGrad;
	neurons[gl_GlobalInvocationID.x].frate -= LEARN_RATE_MEM * biasGrad;
	neurons[gl_GlobalInvocationID.x].mweight -= LEARN_RATE * memGrad;
}