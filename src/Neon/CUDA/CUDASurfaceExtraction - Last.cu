#include <Neon/CUDA/CUDASurfaceExtraction.h>

#include <Neon/NeonScene.h>
#include <Neon/NeonDebugEntity.h>
#include <Neon/Component/NeonMesh.h>
#include <Neon/NeonVertexBufferObject.hpp>

#undef min
#undef max

namespace NeonCUDA
{
	namespace MarchingCubes
	{
		__device__ int edgeTable[256] = {
			0x0  , 0x109, 0x203, 0x30a, 0x406, 0x50f, 0x605, 0x70c,
			0x80c, 0x905, 0xa0f, 0xb06, 0xc0a, 0xd03, 0xe09, 0xf00,
			0x190, 0x99 , 0x393, 0x29a, 0x596, 0x49f, 0x795, 0x69c,
			0x99c, 0x895, 0xb9f, 0xa96, 0xd9a, 0xc93, 0xf99, 0xe90,
			0x230, 0x339, 0x33 , 0x13a, 0x636, 0x73f, 0x435, 0x53c,
			0xa3c, 0xb35, 0x83f, 0x936, 0xe3a, 0xf33, 0xc39, 0xd30,
			0x3a0, 0x2a9, 0x1a3, 0xaa , 0x7a6, 0x6af, 0x5a5, 0x4ac,
			0xbac, 0xaa5, 0x9af, 0x8a6, 0xfaa, 0xea3, 0xda9, 0xca0,
			0x460, 0x569, 0x663, 0x76a, 0x66 , 0x16f, 0x265, 0x36c,
			0xc6c, 0xd65, 0xe6f, 0xf66, 0x86a, 0x963, 0xa69, 0xb60,
			0x5f0, 0x4f9, 0x7f3, 0x6fa, 0x1f6, 0xff , 0x3f5, 0x2fc,
			0xdfc, 0xcf5, 0xfff, 0xef6, 0x9fa, 0x8f3, 0xbf9, 0xaf0,
			0x650, 0x759, 0x453, 0x55a, 0x256, 0x35f, 0x55 , 0x15c,
			0xe5c, 0xf55, 0xc5f, 0xd56, 0xa5a, 0xb53, 0x859, 0x950,
			0x7c0, 0x6c9, 0x5c3, 0x4ca, 0x3c6, 0x2cf, 0x1c5, 0xcc ,
			0xfcc, 0xec5, 0xdcf, 0xcc6, 0xbca, 0xac3, 0x9c9, 0x8c0,
			0x8c0, 0x9c9, 0xac3, 0xbca, 0xcc6, 0xdcf, 0xec5, 0xfcc,
			0xcc , 0x1c5, 0x2cf, 0x3c6, 0x4ca, 0x5c3, 0x6c9, 0x7c0,
			0x950, 0x859, 0xb53, 0xa5a, 0xd56, 0xc5f, 0xf55, 0xe5c,
			0x15c, 0x55 , 0x35f, 0x256, 0x55a, 0x453, 0x759, 0x650,
			0xaf0, 0xbf9, 0x8f3, 0x9fa, 0xef6, 0xfff, 0xcf5, 0xdfc,
			0x2fc, 0x3f5, 0xff , 0x1f6, 0x6fa, 0x7f3, 0x4f9, 0x5f0,
			0xb60, 0xa69, 0x963, 0x86a, 0xf66, 0xe6f, 0xd65, 0xc6c,
			0x36c, 0x265, 0x16f, 0x66 , 0x76a, 0x663, 0x569, 0x460,
			0xca0, 0xda9, 0xea3, 0xfaa, 0x8a6, 0x9af, 0xaa5, 0xbac,
			0x4ac, 0x5a5, 0x6af, 0x7a6, 0xaa , 0x1a3, 0x2a9, 0x3a0,
			0xd30, 0xc39, 0xf33, 0xe3a, 0x936, 0x83f, 0xb35, 0xa3c,
			0x53c, 0x435, 0x73f, 0x636, 0x13a, 0x33 , 0x339, 0x230,
			0xe90, 0xf99, 0xc93, 0xd9a, 0xa96, 0xb9f, 0x895, 0x99c,
			0x69c, 0x795, 0x49f, 0x596, 0x29a, 0x393, 0x99 , 0x190,
			0xf00, 0xe09, 0xd03, 0xc0a, 0xb06, 0xa0f, 0x905, 0x80c,
			0x70c, 0x605, 0x50f, 0x406, 0x30a, 0x203, 0x109, 0x0 };

		__device__ int triTable[256][16] = {
		{-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
		{0, 8, 3, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
		{0, 1, 9, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
		{1, 8, 3, 9, 8, 1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
		{1, 2, 10, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
		{0, 8, 3, 1, 2, 10, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
		{9, 2, 10, 0, 2, 9, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
		{2, 8, 3, 2, 10, 8, 10, 9, 8, -1, -1, -1, -1, -1, -1, -1},
		{3, 11, 2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
		{0, 11, 2, 8, 11, 0, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
		{1, 9, 0, 2, 3, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
		{1, 11, 2, 1, 9, 11, 9, 8, 11, -1, -1, -1, -1, -1, -1, -1},
		{3, 10, 1, 11, 10, 3, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
		{0, 10, 1, 0, 8, 10, 8, 11, 10, -1, -1, -1, -1, -1, -1, -1},
		{3, 9, 0, 3, 11, 9, 11, 10, 9, -1, -1, -1, -1, -1, -1, -1},
		{9, 8, 10, 10, 8, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
		{4, 7, 8, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
		{4, 3, 0, 7, 3, 4, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
		{0, 1, 9, 8, 4, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
		{4, 1, 9, 4, 7, 1, 7, 3, 1, -1, -1, -1, -1, -1, -1, -1},
		{1, 2, 10, 8, 4, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
		{3, 4, 7, 3, 0, 4, 1, 2, 10, -1, -1, -1, -1, -1, -1, -1},
		{9, 2, 10, 9, 0, 2, 8, 4, 7, -1, -1, -1, -1, -1, -1, -1},
		{2, 10, 9, 2, 9, 7, 2, 7, 3, 7, 9, 4, -1, -1, -1, -1},
		{8, 4, 7, 3, 11, 2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
		{11, 4, 7, 11, 2, 4, 2, 0, 4, -1, -1, -1, -1, -1, -1, -1},
		{9, 0, 1, 8, 4, 7, 2, 3, 11, -1, -1, -1, -1, -1, -1, -1},
		{4, 7, 11, 9, 4, 11, 9, 11, 2, 9, 2, 1, -1, -1, -1, -1},
		{3, 10, 1, 3, 11, 10, 7, 8, 4, -1, -1, -1, -1, -1, -1, -1},
		{1, 11, 10, 1, 4, 11, 1, 0, 4, 7, 11, 4, -1, -1, -1, -1},
		{4, 7, 8, 9, 0, 11, 9, 11, 10, 11, 0, 3, -1, -1, -1, -1},
		{4, 7, 11, 4, 11, 9, 9, 11, 10, -1, -1, -1, -1, -1, -1, -1},
		{9, 5, 4, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
		{9, 5, 4, 0, 8, 3, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
		{0, 5, 4, 1, 5, 0, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
		{8, 5, 4, 8, 3, 5, 3, 1, 5, -1, -1, -1, -1, -1, -1, -1},
		{1, 2, 10, 9, 5, 4, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
		{3, 0, 8, 1, 2, 10, 4, 9, 5, -1, -1, -1, -1, -1, -1, -1},
		{5, 2, 10, 5, 4, 2, 4, 0, 2, -1, -1, -1, -1, -1, -1, -1},
		{2, 10, 5, 3, 2, 5, 3, 5, 4, 3, 4, 8, -1, -1, -1, -1},
		{9, 5, 4, 2, 3, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
		{0, 11, 2, 0, 8, 11, 4, 9, 5, -1, -1, -1, -1, -1, -1, -1},
		{0, 5, 4, 0, 1, 5, 2, 3, 11, -1, -1, -1, -1, -1, -1, -1},
		{2, 1, 5, 2, 5, 8, 2, 8, 11, 4, 8, 5, -1, -1, -1, -1},
		{10, 3, 11, 10, 1, 3, 9, 5, 4, -1, -1, -1, -1, -1, -1, -1},
		{4, 9, 5, 0, 8, 1, 8, 10, 1, 8, 11, 10, -1, -1, -1, -1},
		{5, 4, 0, 5, 0, 11, 5, 11, 10, 11, 0, 3, -1, -1, -1, -1},
		{5, 4, 8, 5, 8, 10, 10, 8, 11, -1, -1, -1, -1, -1, -1, -1},
		{9, 7, 8, 5, 7, 9, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
		{9, 3, 0, 9, 5, 3, 5, 7, 3, -1, -1, -1, -1, -1, -1, -1},
		{0, 7, 8, 0, 1, 7, 1, 5, 7, -1, -1, -1, -1, -1, -1, -1},
		{1, 5, 3, 3, 5, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
		{9, 7, 8, 9, 5, 7, 10, 1, 2, -1, -1, -1, -1, -1, -1, -1},
		{10, 1, 2, 9, 5, 0, 5, 3, 0, 5, 7, 3, -1, -1, -1, -1},
		{8, 0, 2, 8, 2, 5, 8, 5, 7, 10, 5, 2, -1, -1, -1, -1},
		{2, 10, 5, 2, 5, 3, 3, 5, 7, -1, -1, -1, -1, -1, -1, -1},
		{7, 9, 5, 7, 8, 9, 3, 11, 2, -1, -1, -1, -1, -1, -1, -1},
		{9, 5, 7, 9, 7, 2, 9, 2, 0, 2, 7, 11, -1, -1, -1, -1},
		{2, 3, 11, 0, 1, 8, 1, 7, 8, 1, 5, 7, -1, -1, -1, -1},
		{11, 2, 1, 11, 1, 7, 7, 1, 5, -1, -1, -1, -1, -1, -1, -1},
		{9, 5, 8, 8, 5, 7, 10, 1, 3, 10, 3, 11, -1, -1, -1, -1},
		{5, 7, 0, 5, 0, 9, 7, 11, 0, 1, 0, 10, 11, 10, 0, -1},
		{11, 10, 0, 11, 0, 3, 10, 5, 0, 8, 0, 7, 5, 7, 0, -1},
		{11, 10, 5, 7, 11, 5, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
		{10, 6, 5, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
		{0, 8, 3, 5, 10, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
		{9, 0, 1, 5, 10, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
		{1, 8, 3, 1, 9, 8, 5, 10, 6, -1, -1, -1, -1, -1, -1, -1},
		{1, 6, 5, 2, 6, 1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
		{1, 6, 5, 1, 2, 6, 3, 0, 8, -1, -1, -1, -1, -1, -1, -1},
		{9, 6, 5, 9, 0, 6, 0, 2, 6, -1, -1, -1, -1, -1, -1, -1},
		{5, 9, 8, 5, 8, 2, 5, 2, 6, 3, 2, 8, -1, -1, -1, -1},
		{2, 3, 11, 10, 6, 5, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
		{11, 0, 8, 11, 2, 0, 10, 6, 5, -1, -1, -1, -1, -1, -1, -1},
		{0, 1, 9, 2, 3, 11, 5, 10, 6, -1, -1, -1, -1, -1, -1, -1},
		{5, 10, 6, 1, 9, 2, 9, 11, 2, 9, 8, 11, -1, -1, -1, -1},
		{6, 3, 11, 6, 5, 3, 5, 1, 3, -1, -1, -1, -1, -1, -1, -1},
		{0, 8, 11, 0, 11, 5, 0, 5, 1, 5, 11, 6, -1, -1, -1, -1},
		{3, 11, 6, 0, 3, 6, 0, 6, 5, 0, 5, 9, -1, -1, -1, -1},
		{6, 5, 9, 6, 9, 11, 11, 9, 8, -1, -1, -1, -1, -1, -1, -1},
		{5, 10, 6, 4, 7, 8, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
		{4, 3, 0, 4, 7, 3, 6, 5, 10, -1, -1, -1, -1, -1, -1, -1},
		{1, 9, 0, 5, 10, 6, 8, 4, 7, -1, -1, -1, -1, -1, -1, -1},
		{10, 6, 5, 1, 9, 7, 1, 7, 3, 7, 9, 4, -1, -1, -1, -1},
		{6, 1, 2, 6, 5, 1, 4, 7, 8, -1, -1, -1, -1, -1, -1, -1},
		{1, 2, 5, 5, 2, 6, 3, 0, 4, 3, 4, 7, -1, -1, -1, -1},
		{8, 4, 7, 9, 0, 5, 0, 6, 5, 0, 2, 6, -1, -1, -1, -1},
		{7, 3, 9, 7, 9, 4, 3, 2, 9, 5, 9, 6, 2, 6, 9, -1},
		{3, 11, 2, 7, 8, 4, 10, 6, 5, -1, -1, -1, -1, -1, -1, -1},
		{5, 10, 6, 4, 7, 2, 4, 2, 0, 2, 7, 11, -1, -1, -1, -1},
		{0, 1, 9, 4, 7, 8, 2, 3, 11, 5, 10, 6, -1, -1, -1, -1},
		{9, 2, 1, 9, 11, 2, 9, 4, 11, 7, 11, 4, 5, 10, 6, -1},
		{8, 4, 7, 3, 11, 5, 3, 5, 1, 5, 11, 6, -1, -1, -1, -1},
		{5, 1, 11, 5, 11, 6, 1, 0, 11, 7, 11, 4, 0, 4, 11, -1},
		{0, 5, 9, 0, 6, 5, 0, 3, 6, 11, 6, 3, 8, 4, 7, -1},
		{6, 5, 9, 6, 9, 11, 4, 7, 9, 7, 11, 9, -1, -1, -1, -1},
		{10, 4, 9, 6, 4, 10, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
		{4, 10, 6, 4, 9, 10, 0, 8, 3, -1, -1, -1, -1, -1, -1, -1},
		{10, 0, 1, 10, 6, 0, 6, 4, 0, -1, -1, -1, -1, -1, -1, -1},
		{8, 3, 1, 8, 1, 6, 8, 6, 4, 6, 1, 10, -1, -1, -1, -1},
		{1, 4, 9, 1, 2, 4, 2, 6, 4, -1, -1, -1, -1, -1, -1, -1},
		{3, 0, 8, 1, 2, 9, 2, 4, 9, 2, 6, 4, -1, -1, -1, -1},
		{0, 2, 4, 4, 2, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
		{8, 3, 2, 8, 2, 4, 4, 2, 6, -1, -1, -1, -1, -1, -1, -1},
		{10, 4, 9, 10, 6, 4, 11, 2, 3, -1, -1, -1, -1, -1, -1, -1},
		{0, 8, 2, 2, 8, 11, 4, 9, 10, 4, 10, 6, -1, -1, -1, -1},
		{3, 11, 2, 0, 1, 6, 0, 6, 4, 6, 1, 10, -1, -1, -1, -1},
		{6, 4, 1, 6, 1, 10, 4, 8, 1, 2, 1, 11, 8, 11, 1, -1},
		{9, 6, 4, 9, 3, 6, 9, 1, 3, 11, 6, 3, -1, -1, -1, -1},
		{8, 11, 1, 8, 1, 0, 11, 6, 1, 9, 1, 4, 6, 4, 1, -1},
		{3, 11, 6, 3, 6, 0, 0, 6, 4, -1, -1, -1, -1, -1, -1, -1},
		{6, 4, 8, 11, 6, 8, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
		{7, 10, 6, 7, 8, 10, 8, 9, 10, -1, -1, -1, -1, -1, -1, -1},
		{0, 7, 3, 0, 10, 7, 0, 9, 10, 6, 7, 10, -1, -1, -1, -1},
		{10, 6, 7, 1, 10, 7, 1, 7, 8, 1, 8, 0, -1, -1, -1, -1},
		{10, 6, 7, 10, 7, 1, 1, 7, 3, -1, -1, -1, -1, -1, -1, -1},
		{1, 2, 6, 1, 6, 8, 1, 8, 9, 8, 6, 7, -1, -1, -1, -1},
		{2, 6, 9, 2, 9, 1, 6, 7, 9, 0, 9, 3, 7, 3, 9, -1},
		{7, 8, 0, 7, 0, 6, 6, 0, 2, -1, -1, -1, -1, -1, -1, -1},
		{7, 3, 2, 6, 7, 2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
		{2, 3, 11, 10, 6, 8, 10, 8, 9, 8, 6, 7, -1, -1, -1, -1},
		{2, 0, 7, 2, 7, 11, 0, 9, 7, 6, 7, 10, 9, 10, 7, -1},
		{1, 8, 0, 1, 7, 8, 1, 10, 7, 6, 7, 10, 2, 3, 11, -1},
		{11, 2, 1, 11, 1, 7, 10, 6, 1, 6, 7, 1, -1, -1, -1, -1},
		{8, 9, 6, 8, 6, 7, 9, 1, 6, 11, 6, 3, 1, 3, 6, -1},
		{0, 9, 1, 11, 6, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
		{7, 8, 0, 7, 0, 6, 3, 11, 0, 11, 6, 0, -1, -1, -1, -1},
		{7, 11, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
		{7, 6, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
		{3, 0, 8, 11, 7, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
		{0, 1, 9, 11, 7, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
		{8, 1, 9, 8, 3, 1, 11, 7, 6, -1, -1, -1, -1, -1, -1, -1},
		{10, 1, 2, 6, 11, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
		{1, 2, 10, 3, 0, 8, 6, 11, 7, -1, -1, -1, -1, -1, -1, -1},
		{2, 9, 0, 2, 10, 9, 6, 11, 7, -1, -1, -1, -1, -1, -1, -1},
		{6, 11, 7, 2, 10, 3, 10, 8, 3, 10, 9, 8, -1, -1, -1, -1},
		{7, 2, 3, 6, 2, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
		{7, 0, 8, 7, 6, 0, 6, 2, 0, -1, -1, -1, -1, -1, -1, -1},
		{2, 7, 6, 2, 3, 7, 0, 1, 9, -1, -1, -1, -1, -1, -1, -1},
		{1, 6, 2, 1, 8, 6, 1, 9, 8, 8, 7, 6, -1, -1, -1, -1},
		{10, 7, 6, 10, 1, 7, 1, 3, 7, -1, -1, -1, -1, -1, -1, -1},
		{10, 7, 6, 1, 7, 10, 1, 8, 7, 1, 0, 8, -1, -1, -1, -1},
		{0, 3, 7, 0, 7, 10, 0, 10, 9, 6, 10, 7, -1, -1, -1, -1},
		{7, 6, 10, 7, 10, 8, 8, 10, 9, -1, -1, -1, -1, -1, -1, -1},
		{6, 8, 4, 11, 8, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
		{3, 6, 11, 3, 0, 6, 0, 4, 6, -1, -1, -1, -1, -1, -1, -1},
		{8, 6, 11, 8, 4, 6, 9, 0, 1, -1, -1, -1, -1, -1, -1, -1},
		{9, 4, 6, 9, 6, 3, 9, 3, 1, 11, 3, 6, -1, -1, -1, -1},
		{6, 8, 4, 6, 11, 8, 2, 10, 1, -1, -1, -1, -1, -1, -1, -1},
		{1, 2, 10, 3, 0, 11, 0, 6, 11, 0, 4, 6, -1, -1, -1, -1},
		{4, 11, 8, 4, 6, 11, 0, 2, 9, 2, 10, 9, -1, -1, -1, -1},
		{10, 9, 3, 10, 3, 2, 9, 4, 3, 11, 3, 6, 4, 6, 3, -1},
		{8, 2, 3, 8, 4, 2, 4, 6, 2, -1, -1, -1, -1, -1, -1, -1},
		{0, 4, 2, 4, 6, 2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
		{1, 9, 0, 2, 3, 4, 2, 4, 6, 4, 3, 8, -1, -1, -1, -1},
		{1, 9, 4, 1, 4, 2, 2, 4, 6, -1, -1, -1, -1, -1, -1, -1},
		{8, 1, 3, 8, 6, 1, 8, 4, 6, 6, 10, 1, -1, -1, -1, -1},
		{10, 1, 0, 10, 0, 6, 6, 0, 4, -1, -1, -1, -1, -1, -1, -1},
		{4, 6, 3, 4, 3, 8, 6, 10, 3, 0, 3, 9, 10, 9, 3, -1},
		{10, 9, 4, 6, 10, 4, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
		{4, 9, 5, 7, 6, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
		{0, 8, 3, 4, 9, 5, 11, 7, 6, -1, -1, -1, -1, -1, -1, -1},
		{5, 0, 1, 5, 4, 0, 7, 6, 11, -1, -1, -1, -1, -1, -1, -1},
		{11, 7, 6, 8, 3, 4, 3, 5, 4, 3, 1, 5, -1, -1, -1, -1},
		{9, 5, 4, 10, 1, 2, 7, 6, 11, -1, -1, -1, -1, -1, -1, -1},
		{6, 11, 7, 1, 2, 10, 0, 8, 3, 4, 9, 5, -1, -1, -1, -1},
		{7, 6, 11, 5, 4, 10, 4, 2, 10, 4, 0, 2, -1, -1, -1, -1},
		{3, 4, 8, 3, 5, 4, 3, 2, 5, 10, 5, 2, 11, 7, 6, -1},
		{7, 2, 3, 7, 6, 2, 5, 4, 9, -1, -1, -1, -1, -1, -1, -1},
		{9, 5, 4, 0, 8, 6, 0, 6, 2, 6, 8, 7, -1, -1, -1, -1},
		{3, 6, 2, 3, 7, 6, 1, 5, 0, 5, 4, 0, -1, -1, -1, -1},
		{6, 2, 8, 6, 8, 7, 2, 1, 8, 4, 8, 5, 1, 5, 8, -1},
		{9, 5, 4, 10, 1, 6, 1, 7, 6, 1, 3, 7, -1, -1, -1, -1},
		{1, 6, 10, 1, 7, 6, 1, 0, 7, 8, 7, 0, 9, 5, 4, -1},
		{4, 0, 10, 4, 10, 5, 0, 3, 10, 6, 10, 7, 3, 7, 10, -1},
		{7, 6, 10, 7, 10, 8, 5, 4, 10, 4, 8, 10, -1, -1, -1, -1},
		{6, 9, 5, 6, 11, 9, 11, 8, 9, -1, -1, -1, -1, -1, -1, -1},
		{3, 6, 11, 0, 6, 3, 0, 5, 6, 0, 9, 5, -1, -1, -1, -1},
		{0, 11, 8, 0, 5, 11, 0, 1, 5, 5, 6, 11, -1, -1, -1, -1},
		{6, 11, 3, 6, 3, 5, 5, 3, 1, -1, -1, -1, -1, -1, -1, -1},
		{1, 2, 10, 9, 5, 11, 9, 11, 8, 11, 5, 6, -1, -1, -1, -1},
		{0, 11, 3, 0, 6, 11, 0, 9, 6, 5, 6, 9, 1, 2, 10, -1},
		{11, 8, 5, 11, 5, 6, 8, 0, 5, 10, 5, 2, 0, 2, 5, -1},
		{6, 11, 3, 6, 3, 5, 2, 10, 3, 10, 5, 3, -1, -1, -1, -1},
		{5, 8, 9, 5, 2, 8, 5, 6, 2, 3, 8, 2, -1, -1, -1, -1},
		{9, 5, 6, 9, 6, 0, 0, 6, 2, -1, -1, -1, -1, -1, -1, -1},
		{1, 5, 8, 1, 8, 0, 5, 6, 8, 3, 8, 2, 6, 2, 8, -1},
		{1, 5, 6, 2, 1, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
		{1, 3, 6, 1, 6, 10, 3, 8, 6, 5, 6, 9, 8, 9, 6, -1},
		{10, 1, 0, 10, 0, 6, 9, 5, 0, 5, 6, 0, -1, -1, -1, -1},
		{0, 3, 8, 5, 6, 10, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
		{10, 5, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
		{11, 5, 10, 7, 5, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
		{11, 5, 10, 11, 7, 5, 8, 3, 0, -1, -1, -1, -1, -1, -1, -1},
		{5, 11, 7, 5, 10, 11, 1, 9, 0, -1, -1, -1, -1, -1, -1, -1},
		{10, 7, 5, 10, 11, 7, 9, 8, 1, 8, 3, 1, -1, -1, -1, -1},
		{11, 1, 2, 11, 7, 1, 7, 5, 1, -1, -1, -1, -1, -1, -1, -1},
		{0, 8, 3, 1, 2, 7, 1, 7, 5, 7, 2, 11, -1, -1, -1, -1},
		{9, 7, 5, 9, 2, 7, 9, 0, 2, 2, 11, 7, -1, -1, -1, -1},
		{7, 5, 2, 7, 2, 11, 5, 9, 2, 3, 2, 8, 9, 8, 2, -1},
		{2, 5, 10, 2, 3, 5, 3, 7, 5, -1, -1, -1, -1, -1, -1, -1},
		{8, 2, 0, 8, 5, 2, 8, 7, 5, 10, 2, 5, -1, -1, -1, -1},
		{9, 0, 1, 5, 10, 3, 5, 3, 7, 3, 10, 2, -1, -1, -1, -1},
		{9, 8, 2, 9, 2, 1, 8, 7, 2, 10, 2, 5, 7, 5, 2, -1},
		{1, 3, 5, 3, 7, 5, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
		{0, 8, 7, 0, 7, 1, 1, 7, 5, -1, -1, -1, -1, -1, -1, -1},
		{9, 0, 3, 9, 3, 5, 5, 3, 7, -1, -1, -1, -1, -1, -1, -1},
		{9, 8, 7, 5, 9, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
		{5, 8, 4, 5, 10, 8, 10, 11, 8, -1, -1, -1, -1, -1, -1, -1},
		{5, 0, 4, 5, 11, 0, 5, 10, 11, 11, 3, 0, -1, -1, -1, -1},
		{0, 1, 9, 8, 4, 10, 8, 10, 11, 10, 4, 5, -1, -1, -1, -1},
		{10, 11, 4, 10, 4, 5, 11, 3, 4, 9, 4, 1, 3, 1, 4, -1},
		{2, 5, 1, 2, 8, 5, 2, 11, 8, 4, 5, 8, -1, -1, -1, -1},
		{0, 4, 11, 0, 11, 3, 4, 5, 11, 2, 11, 1, 5, 1, 11, -1},
		{0, 2, 5, 0, 5, 9, 2, 11, 5, 4, 5, 8, 11, 8, 5, -1},
		{9, 4, 5, 2, 11, 3, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
		{2, 5, 10, 3, 5, 2, 3, 4, 5, 3, 8, 4, -1, -1, -1, -1},
		{5, 10, 2, 5, 2, 4, 4, 2, 0, -1, -1, -1, -1, -1, -1, -1},
		{3, 10, 2, 3, 5, 10, 3, 8, 5, 4, 5, 8, 0, 1, 9, -1},
		{5, 10, 2, 5, 2, 4, 1, 9, 2, 9, 4, 2, -1, -1, -1, -1},
		{8, 4, 5, 8, 5, 3, 3, 5, 1, -1, -1, -1, -1, -1, -1, -1},
		{0, 4, 5, 1, 0, 5, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
		{8, 4, 5, 8, 5, 3, 9, 0, 5, 0, 3, 5, -1, -1, -1, -1},
		{9, 4, 5, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
		{4, 11, 7, 4, 9, 11, 9, 10, 11, -1, -1, -1, -1, -1, -1, -1},
		{0, 8, 3, 4, 9, 7, 9, 11, 7, 9, 10, 11, -1, -1, -1, -1},
		{1, 10, 11, 1, 11, 4, 1, 4, 0, 7, 4, 11, -1, -1, -1, -1},
		{3, 1, 4, 3, 4, 8, 1, 10, 4, 7, 4, 11, 10, 11, 4, -1},
		{4, 11, 7, 9, 11, 4, 9, 2, 11, 9, 1, 2, -1, -1, -1, -1},
		{9, 7, 4, 9, 11, 7, 9, 1, 11, 2, 11, 1, 0, 8, 3, -1},
		{11, 7, 4, 11, 4, 2, 2, 4, 0, -1, -1, -1, -1, -1, -1, -1},
		{11, 7, 4, 11, 4, 2, 8, 3, 4, 3, 2, 4, -1, -1, -1, -1},
		{2, 9, 10, 2, 7, 9, 2, 3, 7, 7, 4, 9, -1, -1, -1, -1},
		{9, 10, 7, 9, 7, 4, 10, 2, 7, 8, 7, 0, 2, 0, 7, -1},
		{3, 7, 10, 3, 10, 2, 7, 4, 10, 1, 10, 0, 4, 0, 10, -1},
		{1, 10, 2, 8, 7, 4, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
		{4, 9, 1, 4, 1, 7, 7, 1, 3, -1, -1, -1, -1, -1, -1, -1},
		{4, 9, 1, 4, 1, 7, 0, 8, 1, 8, 7, 1, -1, -1, -1, -1},
		{4, 0, 3, 7, 4, 3, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
		{4, 8, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
		{9, 10, 8, 10, 11, 8, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
		{3, 0, 9, 3, 9, 11, 11, 9, 10, -1, -1, -1, -1, -1, -1, -1},
		{0, 1, 10, 0, 10, 8, 8, 10, 11, -1, -1, -1, -1, -1, -1, -1},
		{3, 1, 10, 11, 3, 10, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
		{1, 2, 11, 1, 11, 9, 9, 11, 8, -1, -1, -1, -1, -1, -1, -1},
		{3, 0, 9, 3, 9, 11, 1, 2, 9, 2, 11, 9, -1, -1, -1, -1},
		{0, 2, 11, 8, 0, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
		{3, 2, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
		{2, 3, 8, 2, 8, 10, 10, 8, 9, -1, -1, -1, -1, -1, -1, -1},
		{9, 10, 2, 0, 9, 2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
		{2, 3, 8, 2, 8, 10, 0, 1, 8, 1, 10, 8, -1, -1, -1, -1},
		{1, 10, 2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
		{1, 3, 8, 9, 1, 8, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
		{0, 9, 1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
		{0, 3, 8, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
		{-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1} };
	}

#pragma region Math
#define DOT(a, b) (a).x() * (b).x() + (a).y() * (b).y() + (a).z() * (b).z()
#define CROSS(a, b) Eigen::Vector3f((a).y() * (b).z() - (b).y() * (a).z(), (a).z() * (b).x() - (b).z() * (a).x(), (a).x() * (b).y() - (b).x() * (a).y())
#define LENGTHSQUARED(a) DOT((a), (a))
#define LENGTH(a) __fsqrt_rn(LENGTHSQUARED(a))
#define DISTANCESQUARED(a, b) LENGTHSQUARED((a) - (b))
#define DISTANCE(a, b) __fsqrt_rn(DISTANCESQUARED((a), (b)))
#define NORMALIZE(a) (a) / (LENGTH(a))

	//__device__ __host__ Eigen::Vector3f operator+(const Eigen::Vector3f& a, const Eigen::Vector3f& b) {
	//	return Eigen::Vector3f(a.x + b.x, a.y + b.y, a.z + b.z);
	//}
	//__device__ __host__ Eigen::Vector3f operator-(const Eigen::Vector3f& a, const Eigen::Vector3f& b) {
	//	return Eigen::Vector3f(a.x - b.x, a.y - b.y, a.z - b.z);
	//}

	//__device__ __host__ Eigen::Vector3f operator*(float a, const Eigen::Vector3f& b) {
	//	return Eigen::Vector3f(a * b.x, a * b.y, a * b.z);
	//}
	//__device__ __host__ Eigen::Vector3f operator*(const Eigen::Vector3f& a, float b) {
	//	return Eigen::Vector3f(a.x * b, a.y * b, a.z * b);
	//}
	//__device__ __host__ Eigen::Vector3f operator/(const Eigen::Vector3f& a, float b) {
	//	return Eigen::Vector3f(a.x / b, a.y / b, a.z / b);
	//}

	//__device__ __host__ float dot(const Eigen::Vector3f& a, const Eigen::Vector3f& b) {
	//	return a.x * b.x + a.y * b.y + a.z * b.z;
	//}

	//__device__ __host__ Eigen::Vector3f cross(const Eigen::Vector3f& a, const Eigen::Vector3f& b) {
	//	return Eigen::Vector3f((a).y * (b).z - (b).y * (a).z, (a).z * (b).x - (b).z * (a).x, (a).x * (b).y - (b).x * (a).y);
	//}

	//__device__ __host__ float length_squared(const Eigen::Vector3f& a, const Eigen::Vector3f& b) {
	//	return dot(a - b, a - b);
	//}

	//__device__ float length(const Eigen::Vector3f& a, const Eigen::Vector3f& b) {
	//	return __fsqrt_rn(dot(a - b, a - b));
	//}

	//__device__ __host__ float length_squared(const Eigen::Vector3f& a) {
	//	return dot(a, a);
	//}

	//__device__ float length(const Eigen::Vector3f& a) {
	//	return __fsqrt_rn(dot(a, a));
	//}

	//__device__ Eigen::Vector3f normalize(const Eigen::Vector3f& a) {
	//	return a / length(a);
	//}

	//__device__ __host__ float distance_squared(const Eigen::Vector3f& a, const Eigen::Vector3f& b) {
	//	return dot((a - b), (a - b));
	//}

	//__device__ float distance(const Eigen::Vector3f& a, const Eigen::Vector3f& b) {
	//	return __fsqrt_rn(dot((a - b), (a - b)));
	//}

	//__device__
	//	float pointTriangleDistance(const Eigen::Vector3f& point, const Eigen::Vector3f& v0, const Eigen::Vector3f& v1, const Eigen::Vector3f& v2) {
	//	// Calculate the normal of the triangle
	//	Eigen::Vector3f normal = normalize(cross(v1 - v0, v2 - v0));

	//	// Calculate the distance between the point and the plane of the triangle
	//	float dist = dot(point - v0, normal);

	//	// Check if the point is above or below the triangle
	//	Eigen::Vector3f projectedPoint = point - dist * normal;
	//	if (dot(normal, cross(v1 - v0, projectedPoint - v0)) > 0 &&
	//		dot(normal, cross(v2 - v1, projectedPoint - v1)) > 0 &&
	//		dot(normal, cross(v0 - v2, projectedPoint - v2)) > 0) {
	//		// Point is inside the triangle
	//		//return std::abs(distance);
	//		return dist;
	//	}
	//	else {
	//		// Point is outside the triangle
	//		// You can return the distance to the closest edge or vertex if needed
	//		return min(min(distance(point, v0), distance(point, v1)), distance(point, v2));
	//	}
	//}

	bool __device__ ray_triangle_intersect(const Eigen::Vector3f& ray_origin, const Eigen::Vector3f& ray_direction,
		const Eigen::Vector3f& v0, const Eigen::Vector3f& v1, const Eigen::Vector3f& v2, bool enable_backculling, float& distance)
	{
		using Eigen::Vector3f;
		const float epsilon = 1e-7f;

		const Vector3f v0v1 = v1 - v0;
		const Vector3f v0v2 = v2 - v0;

		const Vector3f pvec = ray_direction.cross(v0v2);

		const float det = v0v1.dot(pvec);

		if (enable_backculling)
		{
			// If det is negative, the triangle is back-facing.
			// If det is close to 0, the ray misses the triangle.
			if (det < epsilon)
				return false;
		}
		else
		{
			// If det is close to 0, the ray and triangle are parallel.
			if (std::abs(det) < epsilon)
				return false;
		}
		const float inv_det = 1 / det;

		const Vector3f tvec = ray_origin - v0;
		const auto u = tvec.dot(pvec) * inv_det;
		if (u < 0 || u > 1)
			return false;

		const Vector3f qvec = tvec.cross(v0v1);
		const auto v = ray_direction.dot(qvec) * inv_det;
		if (v < 0 || u + v > 1)
			return false;

		const auto t = v0v2.dot(qvec) * inv_det;

		distance = t;
		return true;
	}
#pragma endregion

	__device__ float atomicCAS_f32(float* p, float cmp, float val) {
		return __int_as_float(atomicCAS((int*)p, __float_as_int(cmp), __float_as_int(val)));
	}

	struct MinMaxFunctor {
		Eigen::Vector3f min_val;
		Eigen::Vector3f max_val;

		MinMaxFunctor() {
			min_val = Eigen::Vector3f(FLT_MAX, FLT_MAX, FLT_MAX);
			max_val = Eigen::Vector3f(-FLT_MAX, -FLT_MAX, -FLT_MAX);
		}

		__host__ __device__
			void operator()(const Eigen::Vector3f& v) {
			if (VECTOR3_VALID(v))
			{
				min_val.x() = min(min_val.x(), v.x());
				min_val.y() = min(min_val.y(), v.y());
				min_val.z() = min(min_val.z(), v.z());

				max_val.x() = max(max_val.x(), v.x());
				max_val.y() = max(max_val.y(), v.y());
				max_val.z() = max(max_val.z(), v.z());
			}
		}
	};

	struct ExtractComponents : public thrust::unary_function<Eigen::Vector3f, float>
	{
		int component;

		ExtractComponents(int component) : component(component) {}

		__host__ __device__
			float operator()(const Eigen::Vector3f& v) const
		{
			return v(component);
		}
	};

	bool __device__ __host__ RayTriangleIntersect(const Eigen::Vector3f& ray_origin, const Eigen::Vector3f& ray_direction,
		const Eigen::Vector3f& v0, const Eigen::Vector3f& v1, const Eigen::Vector3f& v2, bool enable_backculling, float& distance)
	{
		using Eigen::Vector3f;
		const float epsilon = 1e-7f;

		const Vector3f v0v1 = v1 - v0;
		const Vector3f v0v2 = v2 - v0;

		const Vector3f pvec = ray_direction.cross(v0v2);

		const float det = v0v1.dot(pvec);

		if (enable_backculling)
		{
			// If det is negative, the triangle is back-facing.
			// If det is close to 0, the ray misses the triangle.
			if (det < epsilon)
				return false;
		}
		else
		{
			// If det is close to 0, the ray and triangle are parallel.
			if (std::abs(det) < epsilon)
				return false;
		}
		const float inv_det = 1 / det;

		const Vector3f tvec = ray_origin - v0;
		const auto u = tvec.dot(pvec) * inv_det;
		if (u < 0 || u > 1)
			return false;

		const Vector3f qvec = tvec.cross(v0v1);
		const auto v = ray_direction.dot(qvec) * inv_det;
		if (v < 0 || u + v > 1)
			return false;

		const auto t = v0v2.dot(qvec) * inv_det;

		distance = t;
		return true;
	}

	__device__
	float pointToTriangleDistance(const Eigen::Vector3f& point, const Eigen::Vector3f& vertex1, const Eigen::Vector3f& vertex2, const Eigen::Vector3f& vertex3) {
		auto normal = CROSS(vertex2 - vertex1, vertex3 - vertex1);
		normal = NORMALIZE(normal);

		float distance = DOT(point - vertex1, normal) / LENGTH(normal);

		// Project the point onto the plane of the triangle
		auto projection = point - distance * normal;

		// Check if the projection is inside the triangle
		auto edge1 = vertex2 - vertex1;
		auto edge2 = vertex3 - vertex1;
		auto edge3 = vertex3 - vertex2;

		auto normal1 = CROSS(edge1, projection - vertex1);
		auto normal2 = CROSS(edge2, projection - vertex2);
		auto normal3 = CROSS(edge3, projection - vertex3);

		if (DOT(normal1, normal2) >= 0 && DOT(normal1, normal3) >= 0) {
			// The projection is inside the triangle, return the distance
			//return fabsf(distance);
			return distance;
		}
		else {
			// The point is outside the triangle, return the distance to the closest edge
			float minEdgeDistance = min(DISTANCE(point, vertex1), min(DISTANCE(point, vertex2), DISTANCE(point, vertex3)));
			return minEdgeDistance;
		}
	}

	struct BuildGridFunctor
	{
		Eigen::Vector3f center;
		float* values;
		//Eigen::Vector3f* positions;
		//MarchingCubes::GRIDCELL* gridcells;
		MarchingCubes::TRIANGLE* triangles;
		int countX;
		int countY;
		int countZ;
		float minX;
		float minY;
		float minZ;
		float voxelSize;
		float isoValue;
		Eigen::Vector3f direction;
		Eigen::Matrix4f transform;
		bool omitOppositeDirectionFaces = true;

		__device__
			void operator()(size_t index)
		{
			auto z = index / (countX * countY);
			auto y = (index % (countX * countY)) / countX;
			auto x = (index % (countX * countY)) % countX;

			//printf("[%d, %d, %d]\n", x, y, z);
			//return;

			//Eigen::Vector3f position(
			//	minX + x * voxelSize + 0.5f * voxelSize,
			//	minY + y * voxelSize + 0.5f * voxelSize,
			//	minZ + z * voxelSize + 0.5f * voxelSize);

			//printf("Position : %f, %f, %f\n", position.x(), position.y(), position.z());

			MarchingCubes::GRIDCELL gridcell;
			gridcell.p[0] = Eigen::Vector3f(
				minX + x * voxelSize + 0.5f * voxelSize,
				minY + y * voxelSize + 0.5f * voxelSize,
				minZ + z * voxelSize + 0.5f * voxelSize);
			gridcell.p[1] = Eigen::Vector3f(
				minX + (x + 1) * voxelSize + 0.5f * voxelSize,
				minY + y * voxelSize + 0.5f * voxelSize,
				minZ + z * voxelSize + 0.5f * voxelSize);
			gridcell.p[2] = Eigen::Vector3f(
				minX + (x + 1) * voxelSize + 0.5f * voxelSize,
				minY + y * voxelSize + 0.5f * voxelSize,
				minZ + (z + 1) * voxelSize + 0.5f * voxelSize);
			gridcell.p[3] = Eigen::Vector3f(
				minX + x * voxelSize + 0.5f * voxelSize,
				minY + y * voxelSize + 0.5f * voxelSize,
				minZ + (z + 1) * voxelSize + 0.5f * voxelSize);
			gridcell.p[4] = Eigen::Vector3f(
				minX + x * voxelSize + 0.5f * voxelSize,
				minY + (y + 1) * voxelSize + 0.5f * voxelSize,
				minZ + z * voxelSize + 0.5f * voxelSize);
			gridcell.p[5] = Eigen::Vector3f(
				minX + (x + 1) * voxelSize + 0.5f * voxelSize,
				minY + (y + 1) * voxelSize + 0.5f * voxelSize,
				minZ + z * voxelSize + 0.5f * voxelSize);
			gridcell.p[6] = Eigen::Vector3f(
				minX + (x + 1) * voxelSize + 0.5f * voxelSize,
				minY + (y + 1) * voxelSize + 0.5f * voxelSize,
				minZ + (z + 1) * voxelSize + 0.5f * voxelSize);
			gridcell.p[7] = Eigen::Vector3f(
				minX + x * voxelSize + 0.5f * voxelSize,
				minY + (y + 1) * voxelSize + 0.5f * voxelSize,
				minZ + (z + 1) * voxelSize + 0.5f * voxelSize);

			gridcell.val[0] = values[z * countX * countY + y * countX + x];
			gridcell.val[1] = values[z * countX * countY + y * countX + (x + 1)];
			gridcell.val[2] = values[(z + 1) * countX * countY + y * countX + (x + 1)];
			gridcell.val[3] = values[(z + 1) * countX * countY + y * countX + x];
			gridcell.val[4] = values[z * countX * countY + (y + 1) * countX + x];
			gridcell.val[5] = values[z * countX * countY + (y + 1) * countX + (x + 1)];
			gridcell.val[6] = values[(z + 1) * countX * countY + (y + 1) * countX + (x + 1)];
			gridcell.val[7] = values[(z + 1) * countX * countY + (y + 1) * countX + x];

			int cubeindex = 0;
			float isolevel = isoValue;
			Eigen::Vector3f vertlist[12];

			if (gridcell.val[0] < isolevel) cubeindex |= 1;
			if (gridcell.val[1] < isolevel) cubeindex |= 2;
			if (gridcell.val[2] < isolevel) cubeindex |= 4;
			if (gridcell.val[3] < isolevel) cubeindex |= 8;
			if (gridcell.val[4] < isolevel) cubeindex |= 16;
			if (gridcell.val[5] < isolevel) cubeindex |= 32;
			if (gridcell.val[6] < isolevel) cubeindex |= 64;
			if (gridcell.val[7] < isolevel) cubeindex |= 128;

			if (MarchingCubes::edgeTable[cubeindex] == 0)
				return;

			if (MarchingCubes::edgeTable[cubeindex] & 1)
				vertlist[0] =
				VertexInterp(isolevel, gridcell.p[0], gridcell.p[1], gridcell.val[0], gridcell.val[1]);
			if (MarchingCubes::edgeTable[cubeindex] & 2)
				vertlist[1] =
				VertexInterp(isolevel, gridcell.p[1], gridcell.p[2], gridcell.val[1], gridcell.val[2]);
			if (MarchingCubes::edgeTable[cubeindex] & 4)
				vertlist[2] =
				VertexInterp(isolevel, gridcell.p[2], gridcell.p[3], gridcell.val[2], gridcell.val[3]);
			if (MarchingCubes::edgeTable[cubeindex] & 8)
				vertlist[3] =
				VertexInterp(isolevel, gridcell.p[3], gridcell.p[0], gridcell.val[3], gridcell.val[0]);
			if (MarchingCubes::edgeTable[cubeindex] & 16)
				vertlist[4] =
				VertexInterp(isolevel, gridcell.p[4], gridcell.p[5], gridcell.val[4], gridcell.val[5]);
			if (MarchingCubes::edgeTable[cubeindex] & 32)
				vertlist[5] =
				VertexInterp(isolevel, gridcell.p[5], gridcell.p[6], gridcell.val[5], gridcell.val[6]);
			if (MarchingCubes::edgeTable[cubeindex] & 64)
				vertlist[6] =
				VertexInterp(isolevel, gridcell.p[6], gridcell.p[7], gridcell.val[6], gridcell.val[7]);
			if (MarchingCubes::edgeTable[cubeindex] & 128)
				vertlist[7] =
				VertexInterp(isolevel, gridcell.p[7], gridcell.p[4], gridcell.val[7], gridcell.val[4]);
			if (MarchingCubes::edgeTable[cubeindex] & 256)
				vertlist[8] =
				VertexInterp(isolevel, gridcell.p[0], gridcell.p[4], gridcell.val[0], gridcell.val[4]);
			if (MarchingCubes::edgeTable[cubeindex] & 512)
				vertlist[9] =
				VertexInterp(isolevel, gridcell.p[1], gridcell.p[5], gridcell.val[1], gridcell.val[5]);
			if (MarchingCubes::edgeTable[cubeindex] & 1024)
				vertlist[10] =
				VertexInterp(isolevel, gridcell.p[2], gridcell.p[6], gridcell.val[2], gridcell.val[6]);
			if (MarchingCubes::edgeTable[cubeindex] & 2048)
				vertlist[11] =
				VertexInterp(isolevel, gridcell.p[3], gridcell.p[7], gridcell.val[3], gridcell.val[7]);

			MarchingCubes::TRIANGLE tris[4];
			int ntriang = 0;
			for (int i = 0; MarchingCubes::triTable[cubeindex][i] != -1; i += 3) {
				auto v0 = vertlist[MarchingCubes::triTable[cubeindex][i]];
				auto v1 = vertlist[MarchingCubes::triTable[cubeindex][i + 1]];
				auto v2 = vertlist[MarchingCubes::triTable[cubeindex][i + 2]];

				if (omitOppositeDirectionFaces)
				{
					auto normal = NORMALIZE(CROSS(NORMALIZE(v1 - v0), NORMALIZE(v2 - v0)));
					auto dot = DOT(normal, direction);
					if (0 < dot)
					{
						tris[ntriang].p[0] = v0;
						tris[ntriang].p[1] = v1;
						tris[ntriang].p[2] = v2;
						ntriang++;
					}
				}
				else
				{
					tris[ntriang].p[0] = v0;
					tris[ntriang].p[1] = v1;
					tris[ntriang].p[2] = v2;
					ntriang++;
				}
			}

			//if (ntriang != 0)
			//{
			//	printf("ntriang : %d\n", ntriang);
			//}

			for (size_t i = 0; i < ntriang; i++)
			{
				triangles[index * 4 + i] = tris[i];

				//printf("%f, %f, %f\n", tris[i].p->x(), tris[i].p->y(), tris[i].p->z());
			}
		}

		__device__
			Eigen::Vector3f VertexInterp(float isolevel, const Eigen::Vector3f& p1, const Eigen::Vector3f& p2, float valp1, float valp2)
		{
			float mu;
			Eigen::Vector3f p;

			if (fabsf(isolevel - valp1) < 0.00001f)
				return(p1);
			if (fabsf(isolevel - valp2) < 0.00001f)
				return(p2);
			if (fabsf(valp1 - valp2) < 0.00001f)
				return(p1);
			mu = (isolevel - valp1) / (valp2 - valp1);
			p.x() = p1.x() + mu * (p2.x() - p1.x());
			p.y() = p1.y() + mu * (p2.y() - p1.y());
			p.z() = p1.z() + mu * (p2.z() - p1.z());

			return p;
		}
	};

	struct VoxelDepthCalculator
	{
		float* _voxelValues;
		size_t hResolution;
		size_t vResolution;
		float xUnit;
		float yUnit;
		Eigen::AlignedBox3f lastFrameAABB;
		size_t _voxelCountX;
		size_t _voxelCountY;
		size_t _voxelCountZ;
		float _voxelSize;
		Eigen::Vector3f* _voxelCenterPositions;
		Eigen::Matrix4f transform;
		Eigen::Matrix4f inverseTransform;
		const Eigen::Vector3f* _inputPoints;
		GLuint* _meshIndices;

		Eigen::Vector3f* _tempPositions;

		__device__
			void operator()(size_t index)
		{
			Eigen::Vector3f iv;
			{
				auto zIndex = index / (_voxelCountX * _voxelCountY);
				auto yIndex = (index % (_voxelCountX * _voxelCountY)) / _voxelCountX;
				auto xIndex = (index % (_voxelCountX * _voxelCountY)) % _voxelCountX;

				float xpos = lastFrameAABB.min().x() + (float)xIndex * _voxelSize + _voxelSize * 0.5f;
				float ypos = lastFrameAABB.min().y() + (float)yIndex * _voxelSize + _voxelSize * 0.5f;
				float zpos = lastFrameAABB.min().z() + (float)zIndex * _voxelSize + _voxelSize * 0.5f;

				Eigen::Vector4f v(xpos, ypos, zpos, 1.0f);
				auto ivv = inverseTransform * v;
				iv.x() = ivv.x();
				iv.y() = ivv.y();
				iv.z() = ivv.z();

				_tempPositions[index] = iv;
			}
			
			{
				size_t xIndex = (size_t)floorf(iv.x() / xUnit) + hResolution / 2;
				size_t yIndex = (size_t)floorf(iv.y() / yUnit) + vResolution / 2;

				auto i0 = _meshIndices[(yIndex * hResolution + xIndex) * 6 + 0];
				auto i1 = _meshIndices[(yIndex * hResolution + xIndex) * 6 + 1];
				auto i2 = _meshIndices[(yIndex * hResolution + xIndex) * 6 + 2];

				auto i3 = _meshIndices[(yIndex * hResolution + xIndex) * 6 + 3];
				auto i4 = _meshIndices[(yIndex * hResolution + xIndex) * 6 + 4];
				auto i5 = _meshIndices[(yIndex * hResolution + xIndex) * 6 + 5];

				auto& v0 = _inputPoints[i0];
				auto& v1 = _inputPoints[i1];
				auto& v2 = _inputPoints[i2];

				auto& v3 = _inputPoints[i3];
				auto& v4 = _inputPoints[i4];
				auto& v5 = _inputPoints[i5];

				//if (VECTOR3_VALID(v0) && VECTOR3_VALID(v1) && VECTOR3_VALID(v2))
				//{
				//	_voxelValues[index] = 0;
				//}

				//if (VECTOR3_VALID(v3) && VECTOR3_VALID(v4) && VECTOR3_VALID(v5))
				//{
				//	_voxelValues[index] = 0;
				//}

				float distance = FLT_MAX;
				if (RayTriangleIntersect(iv, Eigen::Vector3f(0, 0, 1), v0, v1, v2, false, distance))
				{
					_voxelValues[index] = distance;
				}

				if (RayTriangleIntersect(iv, Eigen::Vector3f(0, 0, 1), v3, v4, v5, false, distance))
				{
					_voxelValues[index] = distance;
				}
			}
		}
	};

	SurfaceExtractor::SurfaceExtractor(size_t hResolution, size_t vResolution, float voxelSize)
		: hResolution(hResolution), vResolution(vResolution), voxelSize(voxelSize)
	{
		Initialize();
	}

	SurfaceExtractor::~SurfaceExtractor()
	{
	}

	void SurfaceExtractor::Initialize()
	{
		depthMap = thrust::device_vector<float>(hResolution * vResolution, FLT_MAX);

		voxelValues = thrust::device_vector<float>(hResolution * vResolution * hResolution, FLT_MAX);

		voxelCenterPositions = thrust::device_vector<Eigen::Vector3f>(hResolution * vResolution * hResolution, Eigen::Vector3f(FLT_MAX, FLT_MAX, FLT_MAX));

		nvtxRangePushA("@Aaron/Build Mesh Indices");
		meshIndices = thrust::device_vector<GLuint>(hResolution * vResolution * 6);
		auto _meshIndices = thrust::raw_pointer_cast(meshIndices.data());

		GLuint _hResolution = (GLuint)hResolution;
		GLuint _vResolution = (GLuint)vResolution;

		thrust::for_each(
			thrust::make_counting_iterator<GLuint>(0),
			thrust::make_counting_iterator<GLuint>((hResolution - 1) * (vResolution - 1)),
			[_meshIndices, _hResolution, _vResolution]__device__(GLuint index) {

			GLuint y = index / _hResolution;
			GLuint x = index % _hResolution;

			if (0 == x % 2 && 0 == y % 3)
			{
				GLuint i0 = _hResolution * y + x;
				GLuint i1 = _hResolution * y + x + 2;
				GLuint i2 = _hResolution * (y + 3) + x;
				GLuint i3 = _hResolution * (y + 3) + x + 2;

				if ((i0 >= _hResolution * _vResolution) ||
					(i1 >= _hResolution * _vResolution) ||
					(i2 >= _hResolution * _vResolution) ||
					(i3 >= _hResolution * _vResolution))
					return;
				
				_meshIndices[index * 6 + 0] = i0;
				_meshIndices[index * 6 + 1] = i1;
				_meshIndices[index * 6 + 2] = i2;

				_meshIndices[index * 6 + 3] = i2;
				_meshIndices[index * 6 + 4] = i1;
				_meshIndices[index * 6 + 5] = i3;
			}
		});
		nvtxRangePop();
	}

	void SurfaceExtractor::PrepareNewFrame()
	{
		nvtxRangePushA("@Aaron/SurfaceExtractor::PrepareNewFrame()");

		thrust::fill(depthMap.begin(), depthMap.end(), FLT_MAX);
		thrust::fill(voxelValues.begin(), voxelValues.end(), -FLT_MAX);
		thrust::fill(voxelCenterPositions.begin(), voxelCenterPositions.end(), Eigen::Vector3f(FLT_MAX, FLT_MAX, FLT_MAX));
		lastFrameAABB.setEmpty();

		nvtxRangePop();
	}

	void SurfaceExtractor::BuildDepthMap(const thrust::device_vector<Eigen::Vector3f>& inputPoints)
	{
		nvtxRangePushA("@Aaron/SurfaceExtractor::BuildDepthMap()");

		auto _depthMap = thrust::raw_pointer_cast(depthMap.data());
		auto _inputPoints = thrust::raw_pointer_cast(inputPoints.data());
		auto _meshIndices = thrust::raw_pointer_cast(meshIndices.data());

		auto _hResolution = hResolution;
		auto _vResolution = vResolution;

		auto _halfWidth = ((float)hResolution * xUnit) * 0.5f;
		auto _halfHeight = ((float)vResolution * yUnit) * 0.5f;

		auto _xUnit = xUnit;
		auto _yUnit = yUnit;
		auto _zUnit = voxelSize;

		//thrust::for_each(
		//	thrust::make_counting_iterator<GLuint>(0),
		//	thrust::make_counting_iterator<GLuint>(hResolution * vResolution * 2),
		//	[_depthMap, _inputPoints, _meshIndices, _hResolution, _vResolution, _halfWidth, _halfHeight,
		//	_xUnit, _yUnit, _zUnit]__device__(GLuint index) {
		//	GLuint i0 = _meshIndices[index * 3 + 0];
		//	GLuint i1 = _meshIndices[index * 3 + 1];
		//	GLuint i2 = _meshIndices[index * 3 + 2];

		//	auto& p0 = _inputPoints[i0];
		//	auto& p1 = _inputPoints[i1];
		//	auto& p2 = _inputPoints[i2];

		//	if (false == VECTOR3_VALID(p0) || false == VECTOR3_VALID(p1) || false == VECTOR3_VALID(p2))
		//		return;

		//	auto aabb = Eigen::AlignedBox3f();
		//	aabb.extend(p0);
		//	aabb.extend(p1);
		//	aabb.extend(p2);

		//	float qminx = floorf(aabb.min().x() / _xUnit) * _xUnit;
		//	float qminy = floorf(aabb.min().y() / _yUnit) * _yUnit;
		//	float qminz = floorf(aabb.min().z() / _zUnit) * _zUnit;

		//	float qmaxx = ceilf(aabb.max().x() / _xUnit) * _xUnit;
		//	float qmaxy = ceilf(aabb.max().y() / _yUnit) * _yUnit;
		//	float qmaxz = ceilf(aabb.max().z() / _zUnit) * _zUnit;

		//	for (float y = qminy; y <= qmaxy; y += _yUnit)
		//	{
		//		for (float x = qminx; x <= qmaxx; x += _xUnit)
		//		{
		//			//printf("%f %f\n", (x + _halfWidth), (y + _halfHeight));

		//			//{
		//			//	auto xIndex = (size_t)floorf((x + _halfWidth) / _xUnit);
		//			//	auto yIndex = (size_t)floorf((y + _halfHeight) / _yUnit);
		//			//	_depthMap[yIndex * _hResolution + xIndex] = 1.0f;
		//			//}
		//			//continue;

		//			float distance = FLT_MAX;
		//			if (RayTriangleIntersect(Eigen::Vector3f(x, y, 0.0f), Eigen::Vector3f(0.0f, 0.0f, 1.0f), p0, p1, p2, false, distance))
		//			{
		//				auto xIndex = (size_t)floorf((x + _halfWidth) / ((float)_hResolution * _xUnit));
		//				auto yIndex = (size_t)floorf((y + _halfHeight) / ((float)_vResolution * _yUnit));

		//				printf("%d %d\n", xIndex, yIndex);

		//				_depthMap[yIndex * _hResolution + xIndex] = distance;
		//			}
		//		}
		//	}
		//});

		nvtxRangePop();

		auto host_inputPoints = thrust::host_vector<Eigen::Vector3f>(inputPoints);
		auto host_meshIndices = thrust::host_vector<GLuint>(meshIndices);
		auto host_depthMap = thrust::host_vector<float>(depthMap);

		cudaDeviceSynchronize();

		//for (size_t index = 0; index < hResolution * vResolution * 2; index++)
		//{
		//	GLuint i0 = host_meshIndices[index * 3 + 0];
		//	GLuint i1 = host_meshIndices[index * 3 + 1];
		//	GLuint i2 = host_meshIndices[index * 3 + 2];

		//	auto& p0 = host_inputPoints[i0];
		//	auto& p1 = host_inputPoints[i1];
		//	auto& p2 = host_inputPoints[i2];

		//	if (false == VECTOR3_VALID(p0) || false == VECTOR3_VALID(p1) || false == VECTOR3_VALID(p2))
		//		continue;

		//	auto aabb = Eigen::AlignedBox3f();
		//	aabb.extend(p0);
		//	aabb.extend(p1);
		//	aabb.extend(p2);

		//	float qminx = floorf((aabb.min().x()) / _xUnit) * _xUnit;
		//	float qminy = floorf((aabb.min().y()) / _yUnit) * _yUnit;

		//	float qmaxx = ceilf((aabb.max().x()) / _xUnit) * _xUnit;
		//	float qmaxy = ceilf((aabb.max().y()) / _yUnit) * _yUnit;

		//	for (float y = qminy; y <= qmaxy; y += _yUnit)
		//	{
		//		for (float x = qminx; x <= qmaxx; x += _xUnit)
		//		{
		//			//printf("%f %f\n", (x + _halfWidth), (y + _halfHeight));

		//			//{
		//			//	auto xIndex = (size_t)floorf((x + _halfWidth) / _xUnit);
		//			//	auto yIndex = (size_t)floorf((y + _halfHeight) / _yUnit);
		//			//	_depthMap[yIndex * _hResolution + xIndex] = 1.0f;
		//			//}
		//			//continue;

		//			float distance = FLT_MAX;
		//			if (RayTriangleIntersect(Eigen::Vector3f(x, y, 0.0f), Eigen::Vector3f(0.0f, 0.0f, 1.0f), p0, p1, p2, false, distance))
		//			{
		//				auto xIndex = (size_t)((x - _halfWidth) / xUnit);
		//				auto yIndex = (size_t)((y - _halfHeight) / yUnit);
		//				
		//				if ((0 <= xIndex && xIndex <= _hResolution - 1) &&
		//					(0 <= yIndex && yIndex <= _vResolution - 1))
		//				{
		//					host_depthMap[yIndex * _hResolution + xIndex] = distance;
		//				}

		//				scene->Debug("Intersection")->AddPoint({ x, y, 0.0f }, glm::red);

		//				scene->Debug("DepthMap_")->AddPoint({ (float)xIndex * xUnit - _halfWidth, (float)yIndex * yUnit - _halfHeight,  0 }, glm::blue);
		//			}
		//		}
		//	}
		//}

		for (size_t y = 0; y < _vResolution; y++)
		{
			for (size_t x = 0; x < _hResolution; x++)
			{
				//host_depthMap[y * _hResolution + x] = 1.0f;

				auto xpos = (float)x * xUnit - _halfWidth;
				auto ypos = (float)y * yUnit - _halfHeight;

				auto xIndex = (size_t)((xpos) / xUnit);
				auto yIndex = (size_t)((ypos) / yUnit);

				scene->Debug("DepthMap")->AddPoint({ xIndex * xUnit - _halfWidth, yIndex * yUnit - _halfHeight,  0 });
				scene->Debug("DepthMap")->AddPoint({ xIndex * xUnit - _halfWidth, yIndex * yUnit - _halfHeight,  0 });
			}
		}
		
		for (size_t y = 0; y <= vResolution; y++)
		{
			for (size_t x = 0; x <= hResolution; x++)
			{
				float z = host_depthMap[y * hResolution + x];
				if (FLT_VALID(z))
				{
					scene->Debug("DepthMap")->AddPoint({ x * xUnit - _halfWidth, y * yUnit - _halfHeight,  0 });
				}
			}
		}
	}

	void SurfaceExtractor::NewFrameWrapper(Neon::Scene* scene, Neon::Mesh* mesh, const Eigen::Matrix4f& transformMatrix)
	{
		this->scene = scene;

		Eigen::AlignedBox3f aabb;

		thrust::host_vector<Eigen::Vector3f> host_meshVertices;
		for (auto& p : mesh->GetVertexBuffer()->GetElements())
		{
			auto v = Eigen::Vector3f(p.x, p.y, p.z);
			host_meshVertices.push_back(v);
			if (VECTOR3_VALID(v))
			{
				aabb.extend(v);
			}
		}
		thrust::device_vector<Eigen::Vector3f> inputPoints(host_meshVertices.begin(), host_meshVertices.end());

#pragma region Draw Trinagles using Input Points
		//{
		//	for (size_t y = 0; y < 480 - 3; y += 3)
		//	{
		//		for (size_t x = 0; x < 256 - 2; x += 2)
		//		{
		//			auto i0 = 256 * y + x;
		//			auto i1 = 256 * y + x + 2;
		//			auto i2 = 256 * (y + 3) + x;
		//			auto i3 = 256 * (y + 3) + x + 2;

		//			auto& p0 = host_meshVertices[i0];
		//			auto& p1 = host_meshVertices[i1];
		//			auto& p2 = host_meshVertices[i2];
		//			auto& p3 = host_meshVertices[i3];

		//			auto v0 = transform * Eigen::Vector4f(p0.x(), p0.y(), p0.z(), 1.0f);
		//			auto v1 = transform * Eigen::Vector4f(p1.x(), p1.y(), p1.z(), 1.0f);
		//			auto v2 = transform * Eigen::Vector4f(p2.x(), p2.y(), p2.z(), 1.0f);
		//			auto v3 = transform * Eigen::Vector4f(p3.x(), p3.y(), p3.z(), 1.0f);

		//			scene->Debug("input triangles1")->AddTriangle(
		//				{ v0.x(), v0.y(), v0.z() },
		//				{ v2.x(), v2.y(), v2.z() },
		//				{ v1.x(), v1.y(), v1.z() },
		//				glm::green, glm::green, glm::green);

		//			scene->Debug("input triangles1")->AddTriangle(
		//				{ v2.x(), v2.y(), v2.z() },
		//				{ v3.x(), v3.y(), v3.z() },
		//				{ v1.x(), v1.y(), v1.z() },
		//				glm::green, glm::green, glm::green);
		//		}
		//	}
		//}
#pragma endregion

#pragma region Draw Triangles using meshIndices
		{
			auto host_meshIndices = thrust::host_vector<GLuint>(meshIndices);
			for (GLuint i = 0; i < (GLuint)hResolution * (GLuint)vResolution * 2; i++)
			{
				GLuint i0 = host_meshIndices[i * 3 + 0];
				GLuint i1 = host_meshIndices[i * 3 + 1];
				GLuint i2 = host_meshIndices[i * 3 + 2];

				auto& p0 = host_meshVertices[i0];
				auto& p1 = host_meshVertices[i1];
				auto& p2 = host_meshVertices[i2];

				if (false == VECTOR3_VALID(p0) || false == VECTOR3_VALID(p1) || false == VECTOR3_VALID(p2))
					continue;

				scene->Debug("input triangles")->AddTriangle(
					{ p0.x(), p0.y(), 0.0f /*p0.z()*/ },
					{ p2.x(), p2.y(), 0.0f /*p2.z()*/ },
					{ p1.x(), p1.y(), 0.0f /*p1.z()*/ },
					glm::green, glm::green, glm::green);
				continue;

				auto tp0 = Eigen::Vector4f(p0.x(), p0.y(), p0.z(), 1.0f);
				auto tp1 = Eigen::Vector4f(p1.x(), p1.y(), p1.z(), 1.0f);
				auto tp2 = Eigen::Vector4f(p2.x(), p2.y(), p2.z(), 1.0f);

				auto tv0 = transformMatrix * tp0;
				auto tv1 = transformMatrix * tp1;
				auto tv2 = transformMatrix * tp2;

				auto v0 = Eigen::Vector3f(tv0.x(), tv0.y(), tv0.z());
				auto v1 = Eigen::Vector3f(tv1.x(), tv1.y(), tv1.z());
				auto v2 = Eigen::Vector3f(tv2.x(), tv2.y(), tv2.z());
								
				scene->Debug("input triangles")->AddTriangle(
					{ v0.x(), v0.y(), v0.z() },
					{ v2.x(), v2.y(), v2.z() },
					{ v1.x(), v1.y(), v1.z() },
					glm::green, glm::green, glm::green);
			}
		}
#pragma endregion

		//float theta = PI / 2.0f;
		//float tx = 10.0f;
		//float ty = 3.0f;
		//float tz = 1.0f;
		//Eigen::Affine3f transformation_matrix = Eigen::Affine3f::Identity();
		//transformation_matrix.rotate(Eigen::AngleAxisf(theta, Eigen::Vector3f::UnitY()));
		//transformation_matrix.translation() << tx, ty, tz;
		//NewFrame(inputPoints, aabb, transformation_matrix.matrix());

		NewFrame(inputPoints, aabb, transformMatrix);

		{
			float isoValue = 0.0f;

			nvtxRangePushA("@Arron/BuildGridCells");

			//auto gridcells = thrust::device_vector<MarchingCubes::GRIDCELL>((voxelCountZ - 1) * (voxelCountY - 1) * (voxelCountX - 1));
			auto triangles = thrust::device_vector<MarchingCubes::TRIANGLE>(voxelCountZ * voxelCountY * voxelCountX * 4);

			auto _voxelValues = thrust::raw_pointer_cast(voxelValues.data());

			//auto _positions = thrust::raw_pointer_cast(positions.data());
			//auto _gridcells = thrust::raw_pointer_cast(gridcells.data());
			auto _triangles = thrust::raw_pointer_cast(triangles.data());

			BuildGridFunctor buildGridFunctor;
			buildGridFunctor.center = Eigen::Vector3f(0.0f, 0.0f, 0.0f);
			buildGridFunctor.values = _voxelValues;
			//buildGridFunctor.gridcells = _gridcells;
			//buildGridFunctor.positions = _positions;
			buildGridFunctor.triangles = _triangles;
			buildGridFunctor.countX = voxelCountX;
			buildGridFunctor.countY = voxelCountY;
			buildGridFunctor.countZ = voxelCountZ;
			buildGridFunctor.minX = lastFrameAABB.center().x() - (float)voxelCountX * voxelSize * 0.5f;
			buildGridFunctor.minY = lastFrameAABB.center().y() - (float)voxelCountY * voxelSize * 0.5f;
			buildGridFunctor.minZ = lastFrameAABB.center().z() - (float)voxelCountZ * voxelSize * 0.5f;
			buildGridFunctor.voxelSize = voxelSize;
			buildGridFunctor.isoValue = isoValue;
			buildGridFunctor.direction = Eigen::Vector3f(transformMatrix.col(2).x(), transformMatrix.col(2).y(), transformMatrix.col(2).z());
			buildGridFunctor.transform = transformMatrix;
			//buildGridFunctor.direction = Eigen::Vector3f(0.0f, 0.0f, 1.0f);
			//buildGridFunctor.omitOppositeDirectionFaces = false;

			printf("Start\n");

			thrust::for_each(thrust::make_counting_iterator<size_t>(0),
				thrust::make_counting_iterator<size_t>((voxelCountX - 1) * (voxelCountY - 1) * (voxelCountZ - 1)),
				buildGridFunctor);

			printf("End\n");

			nvtxRangePop();

			//auto mesh = scene->CreateComponent<Neon::Mesh>("toSave");

			thrust::host_vector<MarchingCubes::TRIANGLE> host_triangles(triangles.begin(), triangles.end());

			for (auto& t : host_triangles)
			{
				if ((t.p[0].x() != 0.0f && t.p[0].y() != 0.0f && t.p[0].z() != 0.0f) &&
					(t.p[1].x() != 0.0f && t.p[1].y() != 0.0f && t.p[1].z() != 0.0f) &&
					(t.p[2].x() != 0.0f && t.p[2].y() != 0.0f && t.p[2].z() != 0.0f))
				{
					scene->Debug("triangles")->AddTriangle(
						{ t.p[0].x(), t.p[0].y(), t.p[0].z() },
						{ t.p[1].x(), t.p[1].y(), t.p[1].z() },
						{ t.p[2].x(), t.p[2].y(), t.p[2].z() });

					
					//auto vi0 = mesh->AddVertex({ t.p[0].x(), t.p[0].y(), t.p[0].z() });
					//auto vi1 = mesh->AddVertex({ t.p[1].x(), t.p[1].y(), t.p[1].z() });
					//auto vi2 = mesh->AddVertex({ t.p[2].x(), t.p[2].y(), t.p[2].z() });

					//mesh->AddTriangle(vi0, vi1, vi2);
				}
			}
			nvtxRangePop();

			//mesh->ToSTLFile("C:\\saveData\\saved.stl");
		}
	}

	void SurfaceExtractor::NewFrame(const thrust::device_vector<Eigen::Vector3f>& inputPoints, const Eigen::AlignedBox3f& aabb, const Eigen::Matrix4f& transformMatrix)
	{
		nvtxRangePushA("@Aaron/SurfaceExtractor::NewFrame()");

		PrepareNewFrame();

		BuildDepthMap(inputPoints);

#pragma region Ready Voxels
		{
			auto transformedAABB = aabb;
			transformedAABB.transform(Eigen::Transform<float, 3, Eigen::Affine>(transformMatrix));

			float xmin = floorf(transformedAABB.min().x() / voxelSize) * voxelSize;
			float ymin = floorf(transformedAABB.min().y() / voxelSize) * voxelSize;
			float zmin = floorf(transformedAABB.min().z() / voxelSize) * voxelSize;

			float xmax = ceilf(transformedAABB.max().x() / voxelSize) * voxelSize;
			float ymax = ceilf(transformedAABB.max().y() / voxelSize) * voxelSize;
			float zmax = ceilf(transformedAABB.max().z() / voxelSize) * voxelSize;

			//scene->Debug("TAABB")->AddAABB(
			//	Neon::AABB(
			//		{ transformedAABB.min().x() , transformedAABB.min().y() , transformedAABB.min().z() },
			//		{ transformedAABB.max().x() , transformedAABB.max().y() , transformedAABB.max().z() }));

			lastFrameAABB = Eigen::AlignedBox3f(Eigen::Vector3f(xmin, ymin, zmin), Eigen::Vector3f(xmax, ymax, zmax));
			voxelCountX = (size_t)((xmax - xmin) / voxelSize);
			voxelCountY = (size_t)((ymax - ymin) / voxelSize);
			voxelCountZ = (size_t)((zmax - zmin) / voxelSize);

			//scene->Debug("TAABB")->AddAABB(
			//	Neon::AABB(
			//		{ lastFrameAABB.min().x() , lastFrameAABB.min().y() , lastFrameAABB.min().z() },
			//		{ lastFrameAABB.max().x() , lastFrameAABB.max().y() , lastFrameAABB.max().z() }));

			auto _lastFrameAABB = lastFrameAABB;
			auto _voxelCountX = voxelCountX;
			auto _voxelCountY = voxelCountY;
			auto _voxelCountZ = voxelCountZ;
			auto _voxelSize = voxelSize;

			auto _voxelCenterPositions = thrust::raw_pointer_cast(voxelCenterPositions.data());

			thrust::for_each(
				thrust::make_counting_iterator<size_t>(0),
				thrust::make_counting_iterator<size_t>(hResolution * vResolution * hResolution),
				[_lastFrameAABB, _voxelCountX, _voxelCountY, _voxelCountZ, _voxelSize, _voxelCenterPositions]
				__device__(size_t index) {

				auto zIndex = index / (_voxelCountX * _voxelCountY);
				auto yIndex = (index % (_voxelCountX * _voxelCountY)) / _voxelCountX;
				auto xIndex = (index % (_voxelCountX * _voxelCountY)) % _voxelCountX;

				float xpos = _lastFrameAABB.min().x() + xIndex * _voxelSize + _voxelSize * 0.5f;
				float ypos = _lastFrameAABB.min().y() + yIndex * _voxelSize + _voxelSize * 0.5f;
				float zpos = _lastFrameAABB.min().z() + zIndex * _voxelSize + _voxelSize * 0.5f;

				_voxelCenterPositions[index].x() = xpos;
				_voxelCenterPositions[index].y() = ypos;
				_voxelCenterPositions[index].z() = zpos;
			});
		}
#pragma endregion

#pragma region Visualize Voxel Center Positions
		//{
		//	auto host_voxelCenterPositions = thrust::host_vector<Eigen::Vector3f>(voxelCenterPositions);

		//	for (size_t i = 0; i < host_voxelCenterPositions.size(); i++)
		//	{
		//		auto& v = host_voxelCenterPositions[i];

		//		if (lastFrameAABB.contains(v))
		//		{
		//			scene->Debug("Voxel Positions")->AddPoint(glm::make_vec3(v.data()), glm::red);
		//		}
		//	}
		//}
#pragma endregion

		{
			auto _inputPoints = thrust::raw_pointer_cast(inputPoints.data());
			auto _meshIndices = thrust::raw_pointer_cast(meshIndices.data());
			auto _voxelValues = thrust::raw_pointer_cast(voxelValues.data());
			auto _transform = transformMatrix;
			auto _inverseTransform = Eigen::Matrix4f(transformMatrix.inverse());

			//scene->Debug("TAABB")->AddAABB(
			//	Neon::AABB(
			//		{ lastFrameAABB.min().x() , lastFrameAABB.min().y() , lastFrameAABB.min().z() },
			//		{ lastFrameAABB.max().x() , lastFrameAABB.max().y() , lastFrameAABB.max().z() }));


			auto _lastFrameAABB = lastFrameAABB;
			auto _voxelCountX = voxelCountX;
			auto _voxelCountY = voxelCountY;
			auto _voxelCountZ = voxelCountZ;
			auto _voxelSize = voxelSize;

			auto _voxelCenterPositions = thrust::raw_pointer_cast(voxelCenterPositions.data());

			{
				thrust::for_each(
					thrust::make_counting_iterator<GLuint>(0),
					thrust::make_counting_iterator<GLuint>(hResolution * vResolution * 2),
					[_inputPoints, _meshIndices, _transform, _voxelValues, _lastFrameAABB, _voxelCountX, _voxelCountY, _voxelCountZ, _voxelSize]
					__device__(GLuint index) {
					GLuint i0 = _meshIndices[index * 3 + 0];
					GLuint i1 = _meshIndices[index * 3 + 1];
					GLuint i2 = _meshIndices[index * 3 + 2];

					auto& p0 = _inputPoints[i0];
					auto& p1 = _inputPoints[i1];
					auto& p2 = _inputPoints[i2];

					auto tv0 = _transform * Eigen::Vector4f(p0.x(), p0.y(), p0.z(), 1.0f);
					auto tv1 = _transform * Eigen::Vector4f(p1.x(), p1.y(), p1.z(), 1.0f);
					auto tv2 = _transform * Eigen::Vector4f(p2.x(), p2.y(), p2.z(), 1.0f);

					auto v0 = Eigen::Vector3f(tv0.x(), tv0.y(), tv0.z());
					auto v1 = Eigen::Vector3f(tv1.x(), tv1.y(), tv1.z());
					auto v2 = Eigen::Vector3f(tv2.x(), tv2.y(), tv2.z());

					if (false == VECTOR3_VALID(v0) || false == VECTOR3_VALID(v1) || false == VECTOR3_VALID(v2))
						return;

					auto aabb = Eigen::AlignedBox3f();
					aabb.extend(v0);
					aabb.extend(v1);
					aabb.extend(v2);

					size_t xminIndex = (size_t)floorf((aabb.min().x() - _lastFrameAABB.min().x()) / _voxelSize);
					size_t yminIndex = (size_t)floorf((aabb.min().y() - _lastFrameAABB.min().y()) / _voxelSize);
					size_t zminIndex = (size_t)floorf((aabb.min().z() - _lastFrameAABB.min().z()) / _voxelSize);

					size_t xmaxIndex = (size_t)ceilf((aabb.max().x() - _lastFrameAABB.min().x()) / _voxelSize);
					size_t ymaxIndex = (size_t)ceilf((aabb.max().y() - _lastFrameAABB.min().y()) / _voxelSize);
					size_t zmaxIndex = (size_t)ceilf((aabb.max().z() - _lastFrameAABB.min().z()) / _voxelSize);


					xminIndex -= 10; if (xminIndex < 0) xminIndex = 0;
					yminIndex -= 10; if (yminIndex < 0) yminIndex = 0;
					zminIndex -= 10; if (zminIndex < 0) zminIndex = 0;

					xmaxIndex += 10; if (xmaxIndex > _voxelCountX - 1) xmaxIndex = _voxelCountX - 1;
					ymaxIndex += 10; if (ymaxIndex > _voxelCountY - 1) ymaxIndex = _voxelCountY - 1;
					zmaxIndex += 10; if (zmaxIndex > _voxelCountZ - 1) zmaxIndex = _voxelCountZ - 1;

					for (size_t z = zminIndex; z < zmaxIndex; z++)
					{
						for (size_t y = yminIndex; y < ymaxIndex; y++)
						{
							for (size_t x = xminIndex; x < xmaxIndex; x++)
							{
								float xpos = _lastFrameAABB.min().x() + x * _voxelSize + _voxelSize * 0.5f;
								float ypos = _lastFrameAABB.min().y() + y * _voxelSize + _voxelSize * 0.5f;
								float zpos = _lastFrameAABB.min().z() + z * _voxelSize + _voxelSize * 0.5f;

								float distance = FLT_MAX;
								auto direction = _transform.col(2);
								//auto direction = Eigen::Vector3f(0.0f, 0.0f, 1.0f);
								if (RayTriangleIntersect(Eigen::Vector3f(xpos, ypos, zpos), Eigen::Vector3f(direction.x(), direction.y(), direction.z()),
									v0, v1, v2, false, distance))
								{
									if (fabsf(distance) < 0.5f)
									{
										if (-distance < _voxelValues[z * (_voxelCountX * _voxelCountY) + y * _voxelCountX + x])
										{
											_voxelValues[z * (_voxelCountX * _voxelCountY) + y * _voxelCountX + x] = -distance;
										}
									}
								}
							}
						}
					}
				});
			}

#if 0
			{
				thrust::for_each(
					thrust::make_counting_iterator<GLuint>(0),
					thrust::make_counting_iterator<GLuint>(hResolution * vResolution * 2),
					[_inputPoints, _meshIndices, _transform, _inverseTransform,
					_voxelValues, _lastFrameAABB, _voxelCountX, _voxelCountY, _voxelCountZ, _voxelSize]
					__device__(GLuint index) {
					GLuint i0 = _meshIndices[index * 3 + 0];
					GLuint i1 = _meshIndices[index * 3 + 1];
					GLuint i2 = _meshIndices[index * 3 + 2];

					auto& p0 = _inputPoints[i0];
					auto& p1 = _inputPoints[i1];
					auto& p2 = _inputPoints[i2];

					auto tv0 = _transform * Eigen::Vector4f(p0.x(), p0.y(), p0.z(), 1.0f);
					auto tv1 = _transform * Eigen::Vector4f(p1.x(), p1.y(), p1.z(), 1.0f);
					auto tv2 = _transform * Eigen::Vector4f(p2.x(), p2.y(), p2.z(), 1.0f);

					auto v0 = Eigen::Vector3f(tv0.x(), tv0.y(), tv0.z());
					auto v1 = Eigen::Vector3f(tv1.x(), tv1.y(), tv1.z());
					auto v2 = Eigen::Vector3f(tv2.x(), tv2.y(), tv2.z());

					if (false == VECTOR3_VALID(v0) || false == VECTOR3_VALID(v1) || false == VECTOR3_VALID(v2))
						return;

					auto aabb = Eigen::AlignedBox3f();
					aabb.extend(v0);
					aabb.extend(v1);
					aabb.extend(v2);

					size_t xminIndex = (size_t)floorf((aabb.min().x() - _lastFrameAABB.min().x()) / _voxelSize);
					size_t yminIndex = (size_t)floorf((aabb.min().y() - _lastFrameAABB.min().y()) / _voxelSize);
					size_t zminIndex = (size_t)floorf((aabb.min().z() - _lastFrameAABB.min().z()) / _voxelSize);

					size_t xmaxIndex = (size_t)ceilf((aabb.max().x() - _lastFrameAABB.min().x()) / _voxelSize);
					size_t ymaxIndex = (size_t)ceilf((aabb.max().y() - _lastFrameAABB.min().y()) / _voxelSize);
					size_t zmaxIndex = (size_t)ceilf((aabb.max().z() - _lastFrameAABB.min().z()) / _voxelSize);

					xminIndex -= 10; if (xminIndex < 0) xminIndex = 0;
					yminIndex -= 10; if (yminIndex < 0) yminIndex = 0;
					zminIndex -= 10; if (zminIndex < 0) zminIndex = 0;

					xmaxIndex += 10; if (xmaxIndex > _voxelCountX - 1) xmaxIndex = _voxelCountX - 1;
					ymaxIndex += 10; if (ymaxIndex > _voxelCountY - 1) ymaxIndex = _voxelCountY - 1;
					zmaxIndex += 10; if (zmaxIndex > _voxelCountZ - 1) zmaxIndex = _voxelCountZ - 1;

					for (size_t z = zminIndex; z < zmaxIndex; z++)
					{
						for (size_t y = yminIndex; y < ymaxIndex; y++)
						{
							for (size_t x = xminIndex; x < xmaxIndex; x++)
							{
								float xpos = _lastFrameAABB.min().x() + (float)x * _voxelSize + _voxelSize * 0.5f;
								float ypos = _lastFrameAABB.min().y() + (float)y * _voxelSize + _voxelSize * 0.5f;
								float zpos = _lastFrameAABB.min().z() + (float)z * _voxelSize + _voxelSize * 0.5f;

								Eigen::Vector4f vc(xpos, ypos, zpos, 1.0f);
								auto ivc = _inverseTransform * vc;

								float distance = FLT_MAX;
								auto direction = Eigen::Vector3f(0.0f, 0.0f, 1.0f);
								if (RayTriangleIntersect(Eigen::Vector3f(ivc.x(), ivc.y(), ivc.z()), Eigen::Vector3f(direction.x(), direction.y(), direction.z()),
									p0, p1, p2, false, distance))
								{
									//float value = distance - zpos;
									float value = zpos - distance;
									//if (fabsf(value) < 1.0f)
									{
										//atomicCAS_f32(&_voxelValues[z * (_voxelCountX * _voxelCountY) + y * _voxelCountX + x], _voxelValues[z * (_voxelCountX * _voxelCountY) + y * _voxelCountX + x], value);

										if (fabsf(value) < fabsf(_voxelValues[z * (_voxelCountX * _voxelCountY) + y * _voxelCountX + x]))
										{
											_voxelValues[z * (_voxelCountX * _voxelCountY) + y * _voxelCountX + x] = value;
											//printf("%f\n", _voxelValues[z * (_voxelCountX * _voxelCountY) + y * _voxelCountX + x]);
										}
									}
								}
							}
						}
					}
				});
			}
#endif // 0


			//auto host_tempPositions = thrust::host_vector<Eigen::Vector3f>(tempPositions);
			//for (size_t i = 0; i < voxelCountX * voxelCountY * voxelCountZ; i++)
			//{
			//	auto& iv = host_tempPositions[i];
			//	if (VECTOR3_VALID(iv))
			//	{
			//		scene->Debug("iv")->AddPoint({ iv.x(), iv.y(), iv.z() }, glm::red);
			//	}
			//}

			//auto host_voxelValues = thrust::host_vector<float>(voxelValues);
			//for (size_t i = 0; i < hResolution * vResolution * hResolution; i++)
			//{
			//	float distance = host_voxelValues[i];
			//	if (FLT_VALID(distance))
			//	{
			//		auto zIndex = i / (_voxelCountX * _voxelCountY);
			//		auto yIndex = (i % (_voxelCountX * _voxelCountY)) / _voxelCountX;
			//		auto xIndex = (i % (_voxelCountX * _voxelCountY)) % _voxelCountX;

			//		float xpos = lastFrameAABB.min().x() + xIndex * voxelSize + voxelSize * 0.5f;
			//		float ypos = lastFrameAABB.min().y() + yIndex * voxelSize + voxelSize * 0.5f;
			//		float zpos = lastFrameAABB.min().z() + zIndex * voxelSize + voxelSize * 0.5f;

			//		if (distance < -0.5f) distance = -0.5f;
			//		if (distance > 0.5f) distance = 0.5f;

			//		float ratio = (distance + 1.0f) / 2.0f;

			//		glm::vec4 c = (1.0f - ratio) * glm::blue + ratio * glm::red;
			//		c.a = 1.0f;

			//		scene->Debug("Mesh Contained Voxels")->AddPoint({ xpos, ypos, zpos }, c);
			//	}
			//}
		}

		nvtxRangePop();
	}
}
