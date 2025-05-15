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
	ACT_DERIV;
}

void main()
{
	uint weightIndex = gl_GlobalInvocationID.x * weightsPerNeuron;
	float outGrad = neurons[gl_GlobalInvocationID.x].grad * ActDeriv(neurons[gl_GlobalInvocationID.x].actout);
	float biasGrad = outGrad + (neurons[gl_GlobalInvocationID.x].bgrad * MOMENTUM);
	
	for (uint w=0; w < weightsPerNeuron; ++w)
	{
		gradients[weightIndex+w] = 
			(outGrad * otherNeurons[w].actout) + 
			(gradients[weightIndex+w] * MOMENTUM);
	}
		
	neurons[gl_GlobalInvocationID.x].grad = outGrad;
	neurons[gl_GlobalInvocationID.x].bgrad = biasGrad;
	
	neurons[gl_GlobalInvocationID.x].bias -= LEARN_RATE_BIAS * biasGrad;
}