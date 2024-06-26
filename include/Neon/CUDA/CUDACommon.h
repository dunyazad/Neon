#pragma once

#include <stdio.h>

#include <algorithm>
#include <cstdlib>

#include <vector>

#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <device_launch_parameters.h>

#include <thrust/iterator/constant_iterator.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/copy.h>
#include <thrust/device_vector.h>
#include <thrust/extrema.h>
#include <thrust/functional.h>
#include <thrust/generate.h>
#include <thrust/host_vector.h>
#include <thrust/reduce.h>
#include <thrust/sort.h>
#include <thrust/tuple.h>
#include <thrust/iterator/zip_iterator.h>

#include <glm/glm.hpp>
#include <glm/gtc/quaternion.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <glm/gtc/matrix_access.hpp>
#include <glm/gtx/intersect.hpp>
#include <glm/gtx/matrix_decompose.hpp>
#include <glm/gtx/quaternion.hpp>
#include <glm/gtx/vector_angle.hpp>

#include <Eigen/Core>
#include <Eigen/Dense>
#include <Eigen/Geometry>
#include <Eigen/LU>

#define FLT_VALID(x) ((x) < FLT_MAX / 2)
//#define FLT_VALID(x) ((x) < 3.402823466e+36F)
#define VECTOR3F_VALID(v) (FLT_VALID((v).x) && FLT_VALID((v).y) && FLT_VALID((v).z))
#define VECTOR3F_VALID_(v) (FLT_VALID((v).x()) && FLT_VALID((v).y()) && FLT_VALID((v).z()))
#define INT_VALID(x) ((x) != INT_MAX)
#define UINT_VALID(x) ((x) != UINT_MAX)
#define VECTOR3U_VALID(v) (UINT_VALID((v).x) && UINT_VALID((v).y) && UINT_VALID((v).z))
#define VECTOR3U_VALID_(v) (UINT_VALID((v).x()) && UINT_VALID((v).y()) && UINT_VALID((v).z()))

#include <vector>

#undef _HAS_STD_BYTE
#define _HAS_STD_BYTE 0
#include <nvtx3/nvToolsExt.h>

#define IntNAN std::numeric_limits<int>::quiet_NaN()
#define RealNAN std::numeric_limits<float>::quiet_NaN()
#define IntInfinity std::numeric_limits<int>::max()

#define DEG2RAD (PI / 180.0f)
#define RAD2DEG (180.0f / PI)

#define CUDA_CHECK_ERROR(err) \
if (err != cudaSuccess && err != cudaErrorUnknown) { \
    fprintf(stderr, "CUDA error: %s at %s:%d\n", cudaGetErrorString(err), __FILE__, __LINE__); \
    exit(err); \
}

namespace NeonCUDA {
	float Trimax(float a, float b, float c);
	float Trimin(float a, float b, float c);

	namespace MarchingCubes
	{
		typedef struct {
			Eigen::Vector3f p[8];
			float val[8];
		} GRIDCELL;

		typedef struct {
			Eigen::Vector3f p[3];
		} TRIANGLE;
	}
}
