#pragma once

#include <Neon/NeonCommon.h>

#include <cuda_runtime.h>

#include <thrust/copy.h>
#include <thrust/device_vector.h>
#include <thrust/generate.h>
#include <thrust/host_vector.h>
#include <thrust/sort.h>
#include <thrust/tuple.h>

namespace NeonCUDA {
	float Trimax(float a, float b, float c);
	float Trimin(float a, float b, float c);
}
