#pragma once
#include <cmath>
#include <cfloat>
#include <algorithm>

struct float2 {
	float x;
	float y;
};

struct float3 {
	float x;
	float y;
	float z;
};

struct float4 {
	float x;
	float y;
	float z;
	float w;
};

struct int2 {
	int32_t x;
	int32_t y;
};

struct int3 {
	int32_t x;
	int32_t y;
	int32_t z;
};

struct int4 {
	int32_t x;
	int32_t y;
	int32_t z;
	int32_t w;
};

struct uint2 {
	uint32_t x;
	uint32_t y;
};

struct uint3 {
	uint32_t x;
	uint32_t y;
	uint32_t z;
};

struct uint4 {
	uint32_t x;
	uint32_t y;
	uint32_t z;
	uint32_t w;
};

template<typename T>
inline void SwapVars(T& x1, T& x2)
{
	T temp = x1;
	x1 = x2;
	x2 = temp;
}

template <typename T>
inline T Clamp(const T& n, const T& lower, const T& upper) {
  return std::max(lower, std::min(n, upper));
}

inline float SmoothStep(float edge0, float edge1, float x) {
    x = Clamp((x - edge0) / (edge1 - edge0), 0.0f, 1.0f);
    return x * x * (3.0f - 2.0f * x);
}

inline double SmoothStep(double edge0, double edge1, double x) {
    x = Clamp((x - edge0) / (edge1 - edge0), 0.0, 1.0);
    return x * x * (3.0 - 2.0 * x);
}

inline uint32_t Index3Dto1D(const uint3& index, const uint32_t& span)
{
	return index.x + span * (index.y + span * index.z);
}

inline uint64_t Index3Dto1D_64(const uint3& index, const uint32_t& span)
{
	return (uint64_t)index.x + span * (index.y + span * index.z);
}

inline uint3 Index1Dto3D(const uint32_t& index, const uint32_t& span)
{
    uint3 result;
    result.x = index % span;
    result.y = (index / span) % span;
    result.z = index / (span * span);
	return result;
}

inline uint2 StrToUInt2(std::string line)
{
	size_t bpos;
	std::string data;
	uint2 result;
	bpos = line.find(",");
	result.x = stoi(line.substr(0, bpos));
	result.y = stoi(line.substr(bpos+1));
	return result;
}

inline uint3 StrToUInt3(std::string line)
{
	size_t bpos;
	std::string data;
	uint3 result;
	bpos = line.find(",");
	result.x = stoi(line.substr(0, bpos));
	data = line.substr(bpos+1);
	bpos = data.find(",");
	result.y = stoi(data.substr(0, bpos));
	result.z = stoi(data.substr(bpos+1));
	return result;
}

inline float2 StrToFlt2(std::string line)
{
	size_t bpos;
	float2 result;
	bpos = line.find(",");
	result.x = stof(line.substr(0, bpos));
	result.y = stof(line.substr(bpos+1));
	return result;
}

inline float3 StrToFlt3(std::string line)
{
	size_t bpos;
	std::string data;
	float3 result;
	bpos = line.find(",");
	result.x = stof(line.substr(0, bpos));
	data = line.substr(bpos+1);
	bpos = data.find(",");
	result.y = stof(data.substr(0, bpos));
	result.z = stof(data.substr(bpos+1));
	return result;
}
