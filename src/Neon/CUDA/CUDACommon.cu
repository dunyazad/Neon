#include "CUDACommon.cuh"

namespace NeonCUDA {

	float Trimax(float a, float b, float c) {
		return std::max(std::max(a, b), c);
	}

	float Trimin(float a, float b, float c) {
		return std::min(std::min(a, b), c);
	}
}