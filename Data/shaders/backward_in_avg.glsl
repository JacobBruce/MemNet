#version 460 core
layout (local_size_x = WORKGROUP_SIZE, local_size_y = 1, local_size_z = 1) in;

NEURON_STRUCT;

layout(std430, binding = 0) buffer LayerBuffer
{
    Neuron neurons[];
};

float ActDeriv(float y)
{
	IN_ACT_DERIV;
}

void main()
{
	float outGrad = neurons[gl_GlobalInvocationID.x].grad;
	
	outGrad = (outGrad == 0.0) ? 
		ActDeriv(neurons[gl_GlobalInvocationID.x].actout) : 
		outGrad * ActDeriv(neurons[gl_GlobalInvocationID.x].actout);
		
	neurons[gl_GlobalInvocationID.x].grad = outGrad;
	neurons[gl_GlobalInvocationID.x].bgrad += outGrad;
	neurons[gl_GlobalInvocationID.x].mgrad += outGrad * neurons[gl_GlobalInvocationID.x].mprev;
}