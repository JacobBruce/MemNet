#version 460 core
layout (local_size_x = WORKGROUP_SIZE, local_size_y = 1, local_size_z = 1) in;

NEURON_STRUCT;

layout(std430, binding = 0) buffer LayerBuffer
{
    Neuron neurons[];
};

float ActDeriv(float y)
{
	ACT_DERIV;
}

void main()
{
	float outGrad = neurons[gl_GlobalInvocationID.x].grad * ActDeriv(neurons[gl_GlobalInvocationID.x].actout);
	float biasGrad = outGrad + (neurons[gl_GlobalInvocationID.x].bgrad * MOMENTUM);
	float memGrad = (outGrad * neurons[gl_GlobalInvocationID.x].mprev) + (neurons[gl_GlobalInvocationID.x].mgrad * MOMENTUM);
		
	neurons[gl_GlobalInvocationID.x].grad = outGrad;
	neurons[gl_GlobalInvocationID.x].bgrad = biasGrad;
	neurons[gl_GlobalInvocationID.x].mgrad = memGrad;
	
	neurons[gl_GlobalInvocationID.x].bias -= LEARN_RATE_BIAS * biasGrad;
	neurons[gl_GlobalInvocationID.x].frate -= LEARN_RATE_MEM * biasGrad;
	neurons[gl_GlobalInvocationID.x].mweight -= LEARN_RATE * memGrad;
}