#include <Neon/CUDA/CUDATSDF.h>

#include <Neon/NeonScene.h>
#include <Neon/NeonDebugEntity.h>
#include <Neon/Component/NeonMesh.h>
#include <Neon/NeonVertexBufferObject.hpp>

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

	__device__
	bool rayTriangleIntersect(
		const Eigen::Vector3f& orig, const Eigen::Vector3f& dir,
		const Eigen::Vector3f& v0, const Eigen::Vector3f& v1, const Eigen::Vector3f& v2, bool enable_backculling,
		float& t)
	{
		const float kEpsilon = 1e-7f;

		// compute the plane's normal
		Eigen::Vector3f v0v1 = v1 - v0;
		Eigen::Vector3f v0v2 = v2 - v0;
		// no need to normalize
		Eigen::Vector3f N = v0v1.cross(v0v2); // N
		float area2 = __fsqrt_rn(N.squaredNorm());

		// Step 1: finding P

		// check if the ray and plane are parallel.
		float NdotRayDirection = N.dot(dir);
		if (fabs(NdotRayDirection) < kEpsilon) // almost 0
			return false; // they are parallel, so they don't intersect! 

		// compute d parameter using equation 2
		float d = -N.dot(v0);

		// compute t (equation 3)
		t = -(N.dot(orig) + d) / NdotRayDirection;

		// check if the triangle is behind the ray
		if(enable_backculling)
			if (t < 0) return false; // the triangle is behind

		// compute the intersection point using equation 1
		Eigen::Vector3f P = orig + t * dir;

		// Step 2: inside-outside test
		Eigen::Vector3f C; // vector perpendicular to triangle's plane

		// edge 0
		Eigen::Vector3f edge0 = v1 - v0;
		Eigen::Vector3f vp0 = P - v0;
		C = edge0.cross(vp0);
		if (N.dot(C) < 0) return false; // P is on the right side

		// edge 1
		Eigen::Vector3f edge1 = v2 - v1;
		Eigen::Vector3f vp1 = P - v1;
		C = edge1.cross(vp1);
		if (N.dot(C) < 0)  return false; // P is on the right side

		// edge 2
		Eigen::Vector3f edge2 = v0 - v2;
		Eigen::Vector3f vp2 = P - v2;
		C = edge2.cross(vp2);
		if (N.dot(C) < 0) return false; // P is on the right side;

		return true; // this ray hits the triangle
	}
#pragma endregion

	struct FillFunctor
	{
		float* values;
		Eigen::Vector3f* positions;
		int countX;
		int countY;
		int countZ;
		float minX;
		float minY;
		float minZ;
		float voxelSize;

		__device__
			void operator()(size_t index)
		{
			auto z = index / (countX * countY);
			auto y = (index % (countX * countY)) / countX;
			auto x = (index % (countX * countY)) % countX;

			Eigen::Vector3f position(
				minX + x * voxelSize + 0.5f * voxelSize,
				minY + y * voxelSize + 0.5f * voxelSize,
				minZ + z * voxelSize + 0.5f * voxelSize);

			values[index] = -FLT_MAX;
			positions[index] = position;
		}
	};

	struct UpdateVoxelsFunctor
	{
		Eigen::Vector3f center;
		float* values;
		Eigen::Vector3f* positions;
		int countX;
		int countY;
		int countZ;
		float minX;
		float minY;
		float minZ;
		float voxelSize;

		__device__
			void operator()(size_t index)
		{
			auto z = index / (countX * countY);
			auto y = (index % (countX * countY)) / countX;
			auto x = (index % (countX * countY)) % countX;

			Eigen::Vector3f position(
				minX + x * voxelSize + 0.5f * voxelSize,
				minY + y * voxelSize + 0.5f * voxelSize,
				minZ + z * voxelSize + 0.5f * voxelSize);

			float dist = __fsqrt_rn((position - center).squaredNorm());

			//if (2.0 < dist)
			//////if (dist > 3.0)
			////if (2.5 < dist)
			//{
			//	values[index] = 1.0f;
			//}
			//else
			//{
			//	values[index] = 0;
			//}

			values[index] = dist;

			//values[index] = position.z;
		}
	};

	struct BuildGridFunctor
	{
		Eigen::Vector3f center;
		float* values;
		//Eigen::Vector3f* positions;
		MarchingCubes::GRIDCELL* gridcells;
		MarchingCubes::TRIANGLE* triangles;
		int countX;
		int countY;
		int countZ;
		float minX;
		float minY;
		float minZ;
		float voxelSize;
		float isoValue;

		__device__
			void operator()(size_t index)
		{
			auto z = index / (countX * countY);
			auto y = (index % (countX * countY)) / countX;
			auto x = (index % (countX * countY)) % countX;

			//printf("[%d, %d, %d]\n", x, y, z);
			//return;

			Eigen::Vector3f position(
				minX + x * voxelSize + 0.5f * voxelSize,
				minY + y * voxelSize + 0.5f * voxelSize,
				minZ + z * voxelSize + 0.5f * voxelSize);

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

				auto normal = NORMALIZE(CROSS(NORMALIZE(v1 - v0), NORMALIZE(v2 - v0)));
				auto zAxis = Eigen::Vector3f(0.0f, 0.0f, 1.0f);
				auto dot = DOT(normal, zAxis);
				if (0 < dot)
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

	struct IntegrateFunctor
	{
		Eigen::Vector3f center;
		float* values;
		Eigen::Vector3f* positions;
		int countX;
		int countY;
		int countZ;
		float minX;
		float minY;
		float minZ;
		float voxelSize;

		const Eigen::Vector3f* inputPositions;
		Eigen::Matrix4f transform;
		Eigen::Matrix4f inverseTransform;
		float width;
		float height;
		int rows;
		int columns;
		float xUnit;
		float yUnit;

		__device__
			void operator()(size_t index)
		{
			auto z = index / (countX * countY);
			auto y = (index % (countX * countY)) / countX;
			auto x = (index % (countX * countY)) % countX;

			Eigen::Vector3f position(
				minX + x * voxelSize + 0.5f * voxelSize,
				minY + y * voxelSize + 0.5f * voxelSize,
				minZ + z * voxelSize + 0.5f * voxelSize);

			Eigen::Vector3f ip = position;

			float xPosition = ip.x() + width * 0.5f;
			float yPosition = ip.y() + height * 0.5f;
			float distance = ip.z();

			if ((0 < xPosition && xPosition < width) &&
				(0 < yPosition && yPosition < height))
			{
				int xIndex = ((int)(xPosition / xUnit) / 2) * 2;
				int yIndex = ((int)(yPosition / yUnit) / 3) * 3;

				auto inputPosition = inputPositions[yIndex * columns + xIndex];

				if (FLT_VALID(inputPosition.x()) && FLT_VALID(inputPosition.y()) && FLT_VALID(inputPosition.z()))
				{
					//printf("[%d, %d] inputPosition : %f, %f, %f\n", xIndex, yIndex, inputPosition.x(), inputPosition.y(), inputPosition.z());
					values[index] = distance - inputPosition.z();

					//printf("%f\n", (inputPosition - Eigen::Vector3f(ip.x(), ip.y(), ip.z())).norm());
				}
				else
				{
					values[index] = FLT_MAX;
					//	//values[index] = 0.0f;
				}
			}
			else
			{
				values[index] = FLT_MAX;
			}
		}
	};

	void Test()
	{

	}

	void DoWork(Neon::Scene* scene, Neon::Mesh* mesh)
	{
		size_t hResolution = 256;
		size_t vResolution = 480;
		float xUnit = 0.1f;
		float yUnit = 0.1f;
		float voxelSize = 0.1f;

		thrust::device_vector<Eigen::Vector3f> depthMap(hResolution * vResolution);

		BuildDepthMap(scene, mesh, hResolution, vResolution, xUnit, yUnit, depthMap);

		size_t voxelCountX = hResolution;
		size_t voxelCountY = vResolution;
		size_t voxelCountZ = hResolution;
		size_t voxelCount = voxelCountX * voxelCountY * voxelCountZ;
		thrust::device_vector<float> voxelValues(voxelCount);
		auto _voxelValues = thrust::raw_pointer_cast(voxelValues.data());
		auto _depthMap = thrust::raw_pointer_cast(depthMap.data());

		thrust::for_each(thrust::make_counting_iterator<size_t>(0), thrust::make_counting_iterator<size_t>(voxelCount),
			[voxelSize, voxelCountX, voxelCountY, voxelCountZ, voxelCount, _voxelValues, _depthMap, hResolution, vResolution]__device__(size_t index) {

			auto z = index / (voxelCountX * voxelCountY);
			auto y = (index % (voxelCountX * voxelCountY)) / voxelCountX;
			auto x = (index % (voxelCountX * voxelCountY)) % voxelCountX;

			float xpos = (float)x * voxelSize - ((float)voxelCountX * voxelSize * 0.5f + voxelSize * 0.5f);
			float ypos = (float)y * voxelSize - ((float)voxelCountY * voxelSize * 0.5f + voxelSize * 0.5f);
			float zpos = (float)z * voxelSize - ((float)voxelCountZ * voxelSize * 0.5f + voxelSize * 0.5f);

			if (FLT_VALID(_depthMap[y * hResolution + x].z()))
			{
				float value = zpos - _depthMap[y * hResolution + x].z();
				if (-0.5f < value && value < 0.5f)
				{
					_voxelValues[index] = value;
				}
				else
				{
					_voxelValues[index] = FLT_MAX;
				}
			}
			else
			{
				_voxelValues[index] = FLT_MAX;
			}
		});

		{
			float isoValue = 0.0f;

			nvtxRangePushA("@Arron/BuildGridCells");

			auto gridcells = thrust::device_vector<MarchingCubes::GRIDCELL>((voxelCountZ - 1) * (voxelCountY - 1) * (voxelCountX - 1));
			auto triangles = thrust::device_vector<MarchingCubes::TRIANGLE>(voxelCountZ * voxelCountY * voxelCountX * 4);

			//auto _positions = thrust::raw_pointer_cast(positions.data());
			auto _gridcells = thrust::raw_pointer_cast(gridcells.data());
			auto _triangles = thrust::raw_pointer_cast(triangles.data());

			BuildGridFunctor buildGridFunctor;
			buildGridFunctor.center = Eigen::Vector3f(0.0f, 0.0f, 0.0f);
			buildGridFunctor.values = _voxelValues;
			buildGridFunctor.gridcells = _gridcells;
			//buildGridFunctor.positions = _positions;
			buildGridFunctor.triangles = _triangles;
			buildGridFunctor.countX = voxelCountX;
			buildGridFunctor.countY = voxelCountY;
			buildGridFunctor.countZ = voxelCountZ;
			buildGridFunctor.minX = -(float)voxelCountX * voxelSize * 0.5f;
			buildGridFunctor.minY = -(float)voxelCountY * voxelSize * 0.5f;
			buildGridFunctor.minZ = -(float)voxelCountZ * voxelSize * 0.5f;
			buildGridFunctor.voxelSize = voxelSize;
			buildGridFunctor.isoValue = isoValue;

			printf("Start\n");

			thrust::for_each(thrust::make_counting_iterator<size_t>(0),
				thrust::make_counting_iterator<size_t>((voxelCountX - 1) * (voxelCountY - 1) * (voxelCountZ - 1)),
				buildGridFunctor);

			printf("End\n");

			nvtxRangePop();

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
				}
			}
		}



		//#pragma region Draw result
		//		{
		//			thrust::host_vector<Eigen::Vector3f> host_result(upscaledDepthMap.begin(), upscaledDepthMap.end());
		//
		//			for (auto& p : host_result)
		//			{
		//				if (VECTOR3_VALID(p))
		//				{
		//					scene->Debug("Upscaling_Result")->AddPoint({ p.x(), p.y(), 0.0f }, glm::yellow);
		//					//scene->Debug("Interpolation_Result_Lines")->AddLine({ p.x(), p.y(), 0.0f }, { p.x(), p.y(), p.z() }, glm::yellow, glm::yellow);
		//				}
		//
		//				if (FLT_VALID(p.x()))
		//				{
		//					float x = floorf(p.x() / xUnit) * xUnit + xUnit * 0.5f;
		//					float y = floorf(p.y() / yUnit) * yUnit + yUnit * 0.5f;
		//					float z = floorf(p.z() / xUnit) * xUnit + xUnit * 0.5f;
		//					//float z = (floorf(p.z() / xUnit) - 1) * xUnit + xUnit * 0.5f;
		//
		//					scene->Debug("Interpolation_Result_Quantized")->AddPoint({ x, y, 0.0f }, glm::red);
		//					//scene->Debug("Interpolation_Result_Lines_Quantized_Lines")->AddLine({ x, y, 0.0f }, { x, y, z }, glm::red, glm::red);
		//				}
		//			}
		//		}
		//#pragma endregion

#pragma region Draw Voxel Points
		{
			thrust::host_vector<float> host_voxelValues(voxelValues.begin(), voxelValues.end());

			for (size_t z = 0; z < voxelCountZ; z++)
			{
				for (size_t y = 0; y < voxelCountY; y++)
				{
					for (size_t x = 0; x < voxelCountX; x++)
					{
						float xpos = (float)x * voxelSize - ((float)voxelCountX * voxelSize * 0.5f + voxelSize * 0.5f);
						float ypos = (float)y * voxelSize - ((float)voxelCountY * voxelSize * 0.5f + voxelSize * 0.5f);
						float zpos = (float)z * voxelSize - ((float)voxelCountZ * voxelSize * 0.5f + voxelSize * 0.5f);

						size_t index = z * voxelCountX * voxelCountY + y * voxelCountX + x;
						auto voxelValue = host_voxelValues[index];

						if (-0.125f <= voxelValue && voxelValue <= 0.125f)
						{
							float ratio = (voxelValue + 0.125f) / 0.25f;

							glm::vec4 c = (1.0f - ratio) * glm::blue + ratio * glm::red;

							scene->Debug("Voxels")->AddPoint({ xpos, ypos, zpos }, c);
						}
					}
				}
			}
		}
#pragma endregion

	}

	std::vector<glm::vec3> FlipXInputArray(const std::vector<glm::vec3>& input, int columns, int rows)
	{
		std::vector<glm::vec3> result;

		for (size_t row = 0; row < rows; row++)
		{
			for (size_t block = (size_t)(columns / 2) - 1; block >= 0; block--)
			{
				size_t index0 = row * columns + block * 2 + 0;
				size_t index1 = row * columns + block * 2 + 1;

				result.push_back(input[index0]);
				result.push_back(input[index1]);

				if (block == 0)
					break;
			}
		}

		return result;
	}

	void BuildDepthMapWrap(Neon::Scene* scene, Neon::Mesh* mesh, size_t hResolution, size_t vResolution, float xUnit, float yUnit)
	{
		thrust::device_vector<Eigen::Vector3f> result(hResolution * vResolution);

		BuildDepthMapNew(scene, mesh, hResolution, vResolution, xUnit, yUnit, result);

		float width = (float)hResolution * xUnit;
		float height = (float)vResolution * yUnit;

		auto voxelSize = 0.05f;

		auto countX = (size_t)ceilf(width / voxelSize);
		auto countY = (size_t)ceilf(height / voxelSize);

		thrust::device_vector<Eigen::Vector3f> upscale_result(countX * countY);
	}

	void BuildDepthMapOld(Neon::Scene* scene, Neon::Mesh* mesh, size_t hResolution, size_t vResolution, float xUnit, float yUnit, thrust::device_vector<Eigen::Vector3f>& result)
	{
		auto inputPositions = mesh->GetVertexBuffer()->GetElements();

		thrust::host_vector<Eigen::Vector3f> host_depthMapInput;
		for (auto& v : inputPositions)
		{
			host_depthMapInput.push_back(Eigen::Vector3f(v.x, v.y, v.z));

			//scene->Debug("DepthMap_Input")->AddPoint({ v.x, v.y, 0.0f });
		}

		thrust::device_vector<Eigen::Vector3f> depthMapInput(host_depthMapInput.begin(), host_depthMapInput.end());

#pragma region Draw Grid
		//{
		//	float minX = -((float)hResolution * xUnit * 0.5f);
		//	float maxX = ((float)hResolution * xUnit * 0.5f);
		//	float minY = -((float)vResolution * yUnit * 0.5f);
		//	float maxY = ((float)vResolution * yUnit * 0.5f);

		//	for (float y = minY; y <= maxY; y += yUnit)
		//	{
		//		scene->Debug("grid lines")->AddLine({ minX, y, 0.0f }, { maxX, y, 0.0f }, glm::lightgray, glm::lightgray);
		//	}
		//	for (float x = minX; x <= maxX; x += xUnit)
		//	{
		//		scene->Debug("grid lines")->AddLine({ x, minY, 0.0f }, { x, maxY, 0.0f }, glm::lightgray, glm::lightgray);
		//	}
		//}
		//{
		//	float minX = -((float)hResolution * xUnit * 0.5f) + xUnit * 0.5f;
		//	float maxX = ((float)hResolution * xUnit * 0.5f) - xUnit * 0.5f;
		//	float minY = -((float)vResolution * yUnit * 0.5f) + yUnit * 0.5f;
		//	float maxY = ((float)vResolution * yUnit * 0.5f) - yUnit * 0.5f;

		//	for (float y = minY; y <= maxY; y += yUnit)
		//	{
		//		scene->Debug("grid lines")->AddLine({ minX, y, 0.0f }, { maxX, y, 0.0f }, glm::gray, glm::gray);
		//	}
		//	for (float x = minX; x <= maxX; x += xUnit)
		//	{
		//		scene->Debug("grid lines")->AddLine({ x, minY, 0.0f }, { x, maxY, 0.0f }, glm::gray, glm::gray);
		//	}
		//}
#pragma endregion

		thrust::fill(result.begin(), result.end(), Eigen::Vector3f(FLT_MAX, FLT_MAX, FLT_MAX));

#pragma region Populate Empty Grid
		auto _result = thrust::raw_pointer_cast(result.data());
		auto _points = thrust::raw_pointer_cast(depthMapInput.data());
		thrust::for_each(thrust::make_counting_iterator<size_t>(0), thrust::make_counting_iterator<size_t>(0) + depthMapInput.size(),
			[_result, _points, hResolution, vResolution, xUnit, yUnit] __device__(size_t index) {

			auto& point = _points[index];
			if (VECTOR3_VALID(point))
			{
				auto xIndex = (int)floorf(point.x() / xUnit) + hResolution / 2;
				auto yIndex = (int)floorf(point.y() / yUnit) + vResolution / 2;

				if (VECTOR3_VALID(_result[yIndex * hResolution + xIndex]))
				{
					auto mean = (point + _result[yIndex * hResolution + xIndex]) * 0.5f;

					//printf("Already exists.\nexists    %4.6f, %4.6f, %4.6f\nto insert %4.6f, %4.6f, %4.6f\nmean      %4.6f, %4.6f, %4.6f\n",
					//	_result[yIndex * hResolution + xIndex].x(), _result[yIndex * hResolution + xIndex].y(), _result[yIndex * hResolution + xIndex].z(),
					//	point.x(), point.y(), point.z(), mean.x(), mean.y(), mean.z());
					_result[yIndex * hResolution + xIndex] = mean;
				}
				else
				{
					_result[yIndex * hResolution + xIndex] = point;
				}

				if (0 == index % 2)
				{
					if (VECTOR3_VALID(_result[yIndex * hResolution + xIndex + 1]))
					{
						auto mean = (point + _result[yIndex * hResolution + xIndex + 1]) * 0.5f;

						//printf("Already exists.\nexists    %4.6f, %4.6f, %4.6f\nto insert %4.6f, %4.6f, %4.6f\nmean      %4.6f, %4.6f, %4.6f\n",
						//	_result[yIndex * hResolution + xIndex].x(), _result[yIndex * hResolution + xIndex].y(), _result[yIndex * hResolution + xIndex].z(),
						//	point.x(), point.y(), point.z(), mean.x(), mean.y(), mean.z());
						_result[yIndex * hResolution + xIndex + 1] = mean;
					}
					else
					{
						auto newPoint = point;
						newPoint.x() += xUnit;
						_result[yIndex * hResolution + xIndex + 1] = newPoint;
					}
				}
			}
		});
#pragma endregion

#pragma region Draw Depthmap Result
		//{
		//	thrust::host_vector<Eigen::Vector3f> host_result(result.begin(), result.end());

		//	for (auto& p : host_result)
		//	{
		//		if (VECTOR3_VALID(p))
		//		{
		//			scene->Debug("DepthMap_Result")->AddPoint({ p.x(), p.y(), 0.0f }, glm::blue);
		//			scene->Debug("DepthMap_Result_Lines")->AddLine({ p.x(), p.y(), 0.0f }, { p.x(), p.y(), p.z() }, glm::blue, glm::blue);
		//		}
		//	}
		//}
#pragma endregion

#pragma region Interpolate and populate Grid
		{
			thrust::for_each(thrust::make_counting_iterator<size_t>(0), thrust::make_counting_iterator<size_t>(0) + depthMapInput.size(),
				[_result, hResolution, vResolution, xUnit, yUnit] __device__(size_t index) {

				auto& p = _result[index];
				if (VECTOR3_VALID(p))
					return;

				int yIndex = (int)(index / hResolution);
				int xIndex = (int)(index % hResolution);

				if (1 < xIndex && xIndex < hResolution - 1)
				{
					auto left = _result[yIndex * hResolution + xIndex - 1];
					if (false == VECTOR3_VALID(left))
					{
						left = _result[yIndex * hResolution + xIndex - 2];
					}

					auto right = _result[yIndex * hResolution + xIndex + 1];
					if (false == VECTOR3_VALID(right))
					{
						right = _result[yIndex * hResolution + xIndex + 2];
					}

					if (VECTOR3_VALID(left) && VECTOR3_VALID(right))
					{
						_result[index] = (left + right) * 0.5f;
					}
					else if (VECTOR3_VALID(left))
					{
						_result[index] = left;
					}
					else if (VECTOR3_VALID(right))
					{
						_result[index] = right;
					}

				}
				else if (xIndex == 0)
				{
					auto right = _result[yIndex * hResolution + xIndex + 1];
					if (false == VECTOR3_VALID(right))
					{
						right = _result[yIndex * hResolution + xIndex + 2];
					}

					if (VECTOR3_VALID(right))
					{
						_result[index] = right;
					}
				}
				else if (xIndex == hResolution - 1)
				{
					auto left = _result[yIndex * hResolution + xIndex - 1];
					if (false == VECTOR3_VALID(left))
					{
						left = _result[yIndex * hResolution + xIndex - 2];
					}

					else if (VECTOR3_VALID(left))
					{
						_result[index] = left;
					}
				}
			});
		}
#pragma endregion
		//
		//#pragma region Fill crack
		//		{
		//			thrust::for_each(thrust::make_counting_iterator<size_t>(0), thrust::make_counting_iterator<size_t>(0) + depthMapInput.size(),
		//				[_result, hResolution, vResolution, xUnit, yUnit] __device__(size_t index) {
		//
		//				auto& p = _result[index];
		//				if (VECTOR3_VALID(p))
		//					return;
		//
		//				int yIndex = (int)(index / hResolution);
		//				int xIndex = (int)(index % hResolution);
		//
		//				_result[index] = Eigen::Vector3f(
		//					xUnit * (float)xIndex -((float)hResolution * xUnit * 0.5f),
		//					yUnit * (float)yIndex -((float)vResolution * yUnit * 0.5f),
		//					-20.f);
		//				return;
		//
		//				auto mean = Eigen::Vector3f(
		//					xUnit * (float)xIndex -((float)hResolution * xUnit * 0.5f),
		//					yUnit * (float)yIndex -((float)vResolution * yUnit * 0.5f),
		//					FLT_MAX);
		//				int validCount = 0;
		//				for (int y = -1; y < 2; y++)
		//				{
		//					for (int x = -1; x < 2; x++)
		//					{
		//						int ix = xIndex + x;
		//						int iy = yIndex + y;
		//
		//						if (ix < 0 || ix > hResolution)
		//							continue;
		//						if (iy < 0 || iy > vResolution)
		//							continue;
		//						if (x == 0 && y == 0)
		//							continue;
		//
		//						auto& np = _result[iy * hResolution + ix];
		//
		//						if (VECTOR3_VALID(np))
		//						{
		//							if (FLT_VALID(mean.z()))
		//							{
		//								mean += np;
		//							}
		//							else
		//							{
		//								mean.x() += np.x();
		//								mean.y() += np.y();
		//								mean.z() = np.z();
		//							}
		//							validCount++;
		//						}
		//					}
		//				}
		//
		//				if (validCount > 0)
		//				{
		//					//_result[index] = Eigen::Vector3f(
		//					//	xUnit * (float)xIndex -((float)hResolution * xUnit * 0.5f),
		//					//	yUnit * (float)yIndex -((float)vResolution * yUnit * 0.5f),
		//					//	-200.f);
		//					//return;
		//
		//					mean /= (float)validCount;
		//					_result[index] = mean;
		//				}
		//			});
		//		}
		//#pragma endregion

#pragma region Draw Result
		//{
		//	thrust::host_vector<Eigen::Vector3f> host_result(result.begin(), result.end());

		//	for (auto& p : host_result)
		//	{
		//		if (VECTOR3_VALID(p))
		//		{
		//			scene->Debug("Interpolation_Result")->AddPoint({ p.x(), p.y(), 0.0f }, glm::yellow);
		//			//scene->Debug("Interpolation_Result_Lines")->AddLine({ p.x(), p.y(), 0.0f }, { p.x(), p.y(), p.z() }, glm::yellow, glm::yellow);
		//		}

		//		if (FLT_VALID(p.x()))
		//		{
		//			float x = floorf(p.x() / xUnit) * xUnit + xUnit * 0.5f;
		//			float y = floorf(p.y() / yUnit) * yUnit + yUnit * 0.5f;
		//			float z = floorf(p.z() / xUnit) * xUnit + xUnit * 0.5f;
		//			//float z = (floorf(p.z() / xUnit) - 1) * xUnit + xUnit * 0.5f;

		//			scene->Debug("Interpolation_Result_Quantized")->AddPoint({ x, y, 0.0f }, glm::red);
		//			//scene->Debug("Interpolation_Result_Lines_Quantized_Lines")->AddLine({ x, y, 0.0f }, { x, y, z }, glm::red, glm::red);
		//		}
		//	}
		//}
#pragma endregion
	}

	void BuildDepthMap(Neon::Scene* scene, Neon::Mesh* mesh, size_t hResolution, size_t vResolution, float xUnit, float yUnit, thrust::device_vector<Eigen::Vector3f>& result)
	{
		auto inputPositions = mesh->GetVertexBuffer()->GetElements();

		thrust::host_vector<Eigen::Vector3f> host_depthMapInput;
		for (auto& v : inputPositions)
		{
			host_depthMapInput.push_back(Eigen::Vector3f(v.x, v.y, v.z));

			//scene->Debug("DepthMap_Input")->AddPoint({ v.x, v.y, 0.0f });
		}

		thrust::device_vector<Eigen::Vector3f> depthMapInput(host_depthMapInput.begin(), host_depthMapInput.end());

		thrust::fill(result.begin(), result.end(), Eigen::Vector3f(FLT_MAX, FLT_MAX, FLT_MAX));

		float width = (float)hResolution * xUnit;
		float height = (float)vResolution * yUnit;

#pragma region Populate Grid
		auto _result = thrust::raw_pointer_cast(result.data());
		auto _points = thrust::raw_pointer_cast(depthMapInput.data());
		thrust::for_each(thrust::make_counting_iterator<size_t>(0), thrust::make_counting_iterator<size_t>(0) + depthMapInput.size(),
			[_result, _points, hResolution, vResolution, xUnit, yUnit] __device__(size_t index) {

			auto& point = _points[index];
			if (VECTOR3_VALID(point))
			{
				auto xIndex = (int)floorf(point.x() / xUnit) + hResolution / 2;
				auto yIndex = (int)floorf(point.y() / yUnit) + vResolution / 2;

				if (VECTOR3_VALID(_result[yIndex * hResolution + xIndex]))
				{
					auto mean = (point + _result[yIndex * hResolution + xIndex]) * 0.5f;

					//printf("Already exists.\nexists    %4.6f, %4.6f, %4.6f\nto insert %4.6f, %4.6f, %4.6f\nmean      %4.6f, %4.6f, %4.6f\n",
					//	_result[yIndex * hResolution + xIndex].x(), _result[yIndex * hResolution + xIndex].y(), _result[yIndex * hResolution + xIndex].z(),
					//	point.x(), point.y(), point.z(), mean.x(), mean.y(), mean.z());
					_result[yIndex * hResolution + xIndex] = mean;
				}
				else
				{
					_result[yIndex * hResolution + xIndex] = point;
				}

				if (0 == index % 2)
				{
					if (VECTOR3_VALID(_result[yIndex * hResolution + xIndex + 1]))
					{
						auto mean = (point + _result[yIndex * hResolution + xIndex + 1]) * 0.5f;

						//printf("Already exists.\nexists    %4.6f, %4.6f, %4.6f\nto insert %4.6f, %4.6f, %4.6f\nmean      %4.6f, %4.6f, %4.6f\n",
						//	_result[yIndex * hResolution + xIndex].x(), _result[yIndex * hResolution + xIndex].y(), _result[yIndex * hResolution + xIndex].z(),
						//	point.x(), point.y(), point.z(), mean.x(), mean.y(), mean.z());
						_result[yIndex * hResolution + xIndex + 1] = mean;
					}
					else
					{
						auto newPoint = point;
						newPoint.x() += xUnit;
						_result[yIndex * hResolution + xIndex + 1] = newPoint;
					}
				}
			}
		});
#pragma endregion

#pragma region Draw Depthmap Result
		//{
		//	thrust::host_vector<Eigen::Vector3f> host_result(result.begin(), result.end());

		//	for (auto& p : host_result)
		//	{
		//		if (VECTOR3_VALID(p))
		//		{
		//			scene->Debug("DepthMap_Result")->AddPoint({ p.x(), p.y(), 0.0f }, glm::blue);
		//			scene->Debug("DepthMap_Result_Lines")->AddLine({ p.x(), p.y(), 0.0f }, { p.x(), p.y(), p.z() }, glm::blue, glm::blue);
		//		}
		//	}
		//}
#pragma endregion

#pragma region Interpolate and populate Grid
		{
			thrust::for_each(thrust::make_counting_iterator<size_t>(0), thrust::make_counting_iterator<size_t>(0) + depthMapInput.size(),
				[_result, hResolution, vResolution, xUnit, yUnit] __device__(size_t index) {

				auto& p = _result[index];
				if (VECTOR3_VALID(p))
					return;

				int yIndex = (int)(index / hResolution);
				int xIndex = (int)(index % hResolution);

				if (1 < xIndex && xIndex < hResolution - 1)
				{
					auto left = _result[yIndex * hResolution + xIndex - 1];
					if (false == VECTOR3_VALID(left))
					{
						left = _result[yIndex * hResolution + xIndex - 2];
					}

					auto right = _result[yIndex * hResolution + xIndex + 1];
					if (false == VECTOR3_VALID(right))
					{
						right = _result[yIndex * hResolution + xIndex + 2];
					}

					if (VECTOR3_VALID(left) && VECTOR3_VALID(right))
					{
						_result[index] = (left + right) * 0.5f;
					}
					else if (VECTOR3_VALID(left))
					{
						_result[index] = left;
					}
					else if (VECTOR3_VALID(right))
					{
						_result[index] = right;
					}

				}
				else if (xIndex == 0)
				{
					auto right = _result[yIndex * hResolution + xIndex + 1];
					if (false == VECTOR3_VALID(right))
					{
						right = _result[yIndex * hResolution + xIndex + 2];
					}

					if (VECTOR3_VALID(right))
					{
						_result[index] = right;
					}
				}
				else if (xIndex == hResolution - 1)
				{
					auto left = _result[yIndex * hResolution + xIndex - 1];
					if (false == VECTOR3_VALID(left))
					{
						left = _result[yIndex * hResolution + xIndex - 2];
					}

					else if (VECTOR3_VALID(left))
					{
						_result[index] = left;
					}
				}
			});
		}
#pragma endregion
		//
		//#pragma region Fill crack
		//		{
		//			thrust::for_each(thrust::make_counting_iterator<size_t>(0), thrust::make_counting_iterator<size_t>(0) + depthMapInput.size(),
		//				[_result, hResolution, vResolution, xUnit, yUnit] __device__(size_t index) {
		//
		//				auto& p = _result[index];
		//				if (VECTOR3_VALID(p))
		//					return;
		//
		//				int yIndex = (int)(index / hResolution);
		//				int xIndex = (int)(index % hResolution);
		//
		//				_result[index] = Eigen::Vector3f(
		//					xUnit * (float)xIndex -((float)hResolution * xUnit * 0.5f),
		//					yUnit * (float)yIndex -((float)vResolution * yUnit * 0.5f),
		//					-20.f);
		//				return;
		//
		//				auto mean = Eigen::Vector3f(
		//					xUnit * (float)xIndex -((float)hResolution * xUnit * 0.5f),
		//					yUnit * (float)yIndex -((float)vResolution * yUnit * 0.5f),
		//					FLT_MAX);
		//				int validCount = 0;
		//				for (int y = -1; y < 2; y++)
		//				{
		//					for (int x = -1; x < 2; x++)
		//					{
		//						int ix = xIndex + x;
		//						int iy = yIndex + y;
		//
		//						if (ix < 0 || ix > hResolution)
		//							continue;
		//						if (iy < 0 || iy > vResolution)
		//							continue;
		//						if (x == 0 && y == 0)
		//							continue;
		//
		//						auto& np = _result[iy * hResolution + ix];
		//
		//						if (VECTOR3_VALID(np))
		//						{
		//							if (FLT_VALID(mean.z()))
		//							{
		//								mean += np;
		//							}
		//							else
		//							{
		//								mean.x() += np.x();
		//								mean.y() += np.y();
		//								mean.z() = np.z();
		//							}
		//							validCount++;
		//						}
		//					}
		//				}
		//
		//				if (validCount > 0)
		//				{
		//					//_result[index] = Eigen::Vector3f(
		//					//	xUnit * (float)xIndex -((float)hResolution * xUnit * 0.5f),
		//					//	yUnit * (float)yIndex -((float)vResolution * yUnit * 0.5f),
		//					//	-200.f);
		//					//return;
		//
		//					mean /= (float)validCount;
		//					_result[index] = mean;
		//				}
		//			});
		//		}
		//#pragma endregion

#pragma region Draw Result
		//{
		//	thrust::host_vector<Eigen::Vector3f> host_result(result.begin(), result.end());

		//	for (auto& p : host_result)
		//	{
		//		if (VECTOR3_VALID(p))
		//		{
		//			scene->Debug("Interpolation_Result")->AddPoint({ p.x(), p.y(), 0.0f }, glm::yellow);
		//			//scene->Debug("Interpolation_Result_Lines")->AddLine({ p.x(), p.y(), 0.0f }, { p.x(), p.y(), p.z() }, glm::yellow, glm::yellow);
		//		}

		//		if (FLT_VALID(p.x()))
		//		{
		//			float x = floorf(p.x() / xUnit) * xUnit + xUnit * 0.5f;
		//			float y = floorf(p.y() / yUnit) * yUnit + yUnit * 0.5f;
		//			float z = floorf(p.z() / xUnit) * xUnit + xUnit * 0.5f;
		//			//float z = (floorf(p.z() / xUnit) - 1) * xUnit + xUnit * 0.5f;

		//			scene->Debug("Interpolation_Result_Quantized")->AddPoint({ x, y, 0.0f }, glm::red);
		//			//scene->Debug("Interpolation_Result_Lines_Quantized_Lines")->AddLine({ x, y, 0.0f }, { x, y, z }, glm::red, glm::red);
		//		}
		//	}
		//}
#pragma endregion
	}

	void BuildDepthMapNew(Neon::Scene* scene, Neon::Mesh* mesh, size_t hResolution, size_t vResolution, float xUnit, float yUnit, thrust::device_vector<Eigen::Vector3f>& result)
	{
		auto inputPositions = mesh->GetVertexBuffer()->GetElements();

#pragma region Mesh Vertices
		thrust::host_vector<Eigen::Vector3f> host_meshVertices;
		for (auto& v : inputPositions)
		{
			host_meshVertices.push_back(Eigen::Vector3f(v.x, v.y, v.z));

			scene->Debug("DepthMap_Input")->AddPoint({ v.x, v.y, 0.0f });
		}

		thrust::device_vector<Eigen::Vector3f> meshVertices(host_meshVertices.begin(), host_meshVertices.end());
		auto _meshVertices = thrust::raw_pointer_cast(meshVertices.data());

		thrust::fill(result.begin(), result.end(), Eigen::Vector3f(FLT_MAX, FLT_MAX, FLT_MAX));
#pragma endregion

#pragma region Mesh Indices
		thrust::device_vector<GLuint> meshIndices(hResolution * vResolution * 6);
		auto _meshIndices = thrust::raw_pointer_cast(meshIndices.data());

		thrust::for_each(
			thrust::make_counting_iterator<GLuint>(0),
			thrust::make_counting_iterator<GLuint>((hResolution - 1) * (vResolution - 1)),
			[_meshIndices, hResolution, vResolution]__device__(size_t index) {

			auto y = index / hResolution;
			auto x = index % hResolution;

			if (0 == x % 2 && 0 == y % 3)
			{
				auto i0 = hResolution * y + x;
				auto i1 = hResolution * y + x + 2;
				auto i2 = hResolution * (y + 3) + x;
				auto i3 = hResolution * (y + 3) + x + 2;

				if ((i0 >= hResolution * vResolution) ||
					(i1 >= hResolution * vResolution) ||
					(i2 >= hResolution * vResolution) ||
					(i3 >= hResolution * vResolution))
					return;

				_meshIndices[index * 6 + 0] = i0;
				_meshIndices[index * 6 + 1] = i1;
				_meshIndices[index * 6 + 2] = i2;

				_meshIndices[index * 6 + 3] = i2;
				_meshIndices[index * 6 + 4] = i1;
				_meshIndices[index * 6 + 5] = i3;
			}
		});
#pragma endregion

#pragma region Draw Mesh
		thrust::host_vector<GLuint> host_meshIndices(meshIndices);

		for (size_t i = 0; i < hResolution * vResolution * 2; i++)
		{
			auto i0 = host_meshIndices[i * 3 + 0];
			auto i1 = host_meshIndices[i * 3 + 1];
			auto i2 = host_meshIndices[i * 3 + 2];

			auto v0 = inputPositions[i0];
			auto v1 = inputPositions[i1];
			auto v2 = inputPositions[i2];

			if (VEC3_VALID(v0) && VEC3_VALID(v1) && VEC3_VALID(v2))
			{
				v0.z = 0;
				v1.z = 0;
				v2.z = 0;

				scene->Debug("Meshing Result")->AddTriangle(v0, v2, v1, glm::green, glm::green, glm::green);
			}
		}
#pragma endregion

		thrust::device_vector<float> depthMap((size_t)((float)hResolution / xUnit) * (size_t)((float)vResolution / yUnit));
		auto _depthMap = thrust::raw_pointer_cast(depthMap.data());

		float _xMin = -(hResolution * xUnit) * 0.5f;
		float _yMin = -(vResolution * yUnit) * 0.5f;

#pragma region Fill Depth Map
		thrust::for_each(
			thrust::make_counting_iterator<size_t>(0),
			thrust::make_counting_iterator<size_t>(hResolution * vResolution * 2),
			[_meshVertices, _meshIndices, hResolution, vResolution, _depthMap, _xMin, _yMin, xUnit, yUnit]
		__device__(size_t index) {
			auto i0 = _meshIndices[index * 3 + 0];
			auto i1 = _meshIndices[index * 3 + 1];
			auto i2 = _meshIndices[index * 3 + 2];

			auto& v0 = _meshVertices[i0];
			auto& v1 = _meshVertices[i1];
			auto& v2 = _meshVertices[i2];

			if (false == VECTOR3_VALID(v0) || false == VECTOR3_VALID(v1) || false == VECTOR3_VALID(v2))
				return;

			float minX = FLT_MAX; minX = min(minX, v0.x()); minX = min(minX, v1.x()); minX = min(minX, v2.x());
			float minY = FLT_MAX; minY = min(minY, v0.y()); minY = min(minY, v1.y()); minY = min(minY, v2.y());
			float maxX = -FLT_MAX; maxX = max(maxX, v0.x()); maxX = max(maxX, v1.x()); maxX = max(maxX, v2.x());
			float maxY = -FLT_MAX; maxY = max(maxY, v0.y()); maxY = max(maxY, v1.y()); maxY = max(maxY, v2.y());

			float aabbMinX = floorf(minX / xUnit) * xUnit;
			float aabbMinY = floorf(minY / yUnit) * yUnit;

			float aabbMaxX = ceilf(maxX / xUnit) * xUnit;
			float aabbMaxY = ceilf(maxY / yUnit) * yUnit;

			int cnt = 0;
			for (float y = aabbMinY; y <= aabbMaxY; y += yUnit)
			{
				for (float x = aabbMinX; x <= aabbMaxX; x += xUnit)
				{
					float distance = 0.0f;
					if (ray_triangle_intersect({ x + xUnit * 0.5f, y + yUnit * 0.5f, 0.0f }, { 0.0f, 0.0f, 1.0f }, v0, v1, v2, false, distance))
						//if (rayTriangleIntersect({ x, y, 0.0f }, { 0.0f, 0.0f, 1.0f }, v0, v1, v2, false, distance))
					{
						auto xIndex = (size_t)floorf((x + xUnit * 0.5f - _xMin) / xUnit);
						auto yIndex = (size_t)floorf((y + yUnit * 0.5f - _yMin) / yUnit);

						_depthMap[yIndex * hResolution + xIndex] = distance;
						cnt++;
					}
				}
			}

			if (cnt == 0)
			{
				printf("xmin: %f xmax: %f ymin: %f ymax: %f dx: %f dy: %f\n",
					aabbMinX, aabbMaxX, aabbMinY, aabbMaxY, aabbMaxX - aabbMinX, aabbMaxY - aabbMinY);

				/*
				for (float y = aabbMinY; y <= aabbMaxY; y += yUnit)
				{
					for (float x = aabbMinX; x <= aabbMaxX; x += xUnit)
					{
						auto xIndex = (size_t)floorf((x - _xMin) / xUnit);
						auto yIndex = (size_t)floorf((y - _yMin) / yUnit);
						_depthMap[yIndex * hResolution + xIndex] = -100.0f;
					}
				}
				*/
			}
		});
#pragma endregion

		thrust::host_vector<float> host_depthMap(depthMap);
		for (size_t i = 0; i < hResolution * vResolution; i++)
		{
			auto yIndex = i / hResolution;
			auto xIndex = i % hResolution;

			auto x = _xMin + xIndex * xUnit + xUnit * 0.5f;
			auto y = _yMin + yIndex * yUnit + yUnit * 0.5f;
			auto z = host_depthMap[i];

			if (FLT_VALID(z) && 0 < fabsf(z))
			{
				//scene->Debug("Depth Map Result with Line")->AddLine({x, y, 0.0f}, {x, y, z}, glm::red, glm::red);
				scene->Debug("Depth Map Result with Point")->AddPoint({ x, y, 0.0f }, glm::red);
			}
			else if (-101.0f < z && z < -99.0f)
			{
				scene->Debug("Depth Map Result with Error Point")->AddPoint({ x, y, 1.0f }, glm::blue);
			}
			else
			{
				//scene->Debug("Depth Map Result with Point")->AddPoint({ x, y, 0.0f }, glm::red);
				//scene->Debug("Depth Map Result with Point")->AddPoint({ x, y, 0.0f }, glm::green);
			}
		}

		return;



		float width = (float)hResolution * xUnit;
		float height = (float)vResolution * yUnit;

#pragma region Populate Grid
		auto _result = thrust::raw_pointer_cast(result.data());
		auto _points = thrust::raw_pointer_cast(meshVertices.data());
		thrust::for_each(
			thrust::make_counting_iterator<size_t>(0),
			thrust::make_counting_iterator<size_t>(0) + meshVertices.size(),
			[_result, _points, hResolution, vResolution, xUnit, yUnit] __device__(size_t index) {

			auto& point = _points[index];
			if (VECTOR3_VALID(point))
			{
				auto xIndex = (int)floorf(point.x() / xUnit) + hResolution / 2;
				auto yIndex = (int)floorf(point.y() / yUnit) + vResolution / 2;

				if (VECTOR3_VALID(_result[yIndex * hResolution + xIndex]))
				{
					auto mean = (point + _result[yIndex * hResolution + xIndex]) * 0.5f;

					//printf("Already exists.\nexists    %4.6f, %4.6f, %4.6f\nto insert %4.6f, %4.6f, %4.6f\nmean      %4.6f, %4.6f, %4.6f\n",
					//	_result[yIndex * hResolution + xIndex].x(), _result[yIndex * hResolution + xIndex].y(), _result[yIndex * hResolution + xIndex].z(),
					//	point.x(), point.y(), point.z(), mean.x(), mean.y(), mean.z());
					_result[yIndex * hResolution + xIndex] = mean;
				}
				else
				{
					_result[yIndex * hResolution + xIndex] = point;
				}

				if (0 == index % 2)
				{
					if (VECTOR3_VALID(_result[yIndex * hResolution + xIndex + 1]))
					{
						auto mean = (point + _result[yIndex * hResolution + xIndex + 1]) * 0.5f;

						//printf("Already exists.\nexists    %4.6f, %4.6f, %4.6f\nto insert %4.6f, %4.6f, %4.6f\nmean      %4.6f, %4.6f, %4.6f\n",
						//	_result[yIndex * hResolution + xIndex].x(), _result[yIndex * hResolution + xIndex].y(), _result[yIndex * hResolution + xIndex].z(),
						//	point.x(), point.y(), point.z(), mean.x(), mean.y(), mean.z());
						_result[yIndex * hResolution + xIndex + 1] = mean;
					}
					else
					{
						auto newPoint = point;
						newPoint.x() += xUnit;
						_result[yIndex * hResolution + xIndex + 1] = newPoint;
					}
				}
			}
		});
#pragma endregion

#pragma region Draw Depthmap Result
		{
			thrust::host_vector<Eigen::Vector3f> host_result(result.begin(), result.end());

			for (auto& p : host_result)
			{
				if (VECTOR3_VALID(p))
				{
					scene->Debug("DepthMap_Result")->AddPoint({ p.x(), p.y(), 0.0f }, glm::blue);
					//scene->Debug("DepthMap_Result_Lines")->AddLine({ p.x(), p.y(), 0.0f }, { p.x(), p.y(), p.z() }, glm::blue, glm::blue);
				}
			}
		}
#pragma endregion
		return;

#pragma region Interpolate and populate Grid
		{
			thrust::for_each(thrust::make_counting_iterator<size_t>(0), thrust::make_counting_iterator<size_t>(0) + meshVertices.size(),
				[_result, hResolution, vResolution, xUnit, yUnit] __device__(size_t index) {

				auto& p = _result[index];
				if (VECTOR3_VALID(p))
					return;

				int yIndex = (int)(index / hResolution);
				int xIndex = (int)(index % hResolution);

				if (1 < xIndex && xIndex < hResolution - 1)
				{
					auto left = _result[yIndex * hResolution + xIndex - 1];
					if (false == VECTOR3_VALID(left))
					{
						left = _result[yIndex * hResolution + xIndex - 2];
					}

					auto right = _result[yIndex * hResolution + xIndex + 1];
					if (false == VECTOR3_VALID(right))
					{
						right = _result[yIndex * hResolution + xIndex + 2];
					}

					if (VECTOR3_VALID(left) && VECTOR3_VALID(right))
					{
						_result[index] = (left + right) * 0.5f;
					}
					else if (VECTOR3_VALID(left))
					{
						_result[index] = left;
					}
					else if (VECTOR3_VALID(right))
					{
						_result[index] = right;
					}

				}
				else if (xIndex == 0)
				{
					auto right = _result[yIndex * hResolution + xIndex + 1];
					if (false == VECTOR3_VALID(right))
					{
						right = _result[yIndex * hResolution + xIndex + 2];
					}

					if (VECTOR3_VALID(right))
					{
						_result[index] = right;
					}
				}
				else if (xIndex == hResolution - 1)
				{
					auto left = _result[yIndex * hResolution + xIndex - 1];
					if (false == VECTOR3_VALID(left))
					{
						left = _result[yIndex * hResolution + xIndex - 2];
					}

					else if (VECTOR3_VALID(left))
					{
						_result[index] = left;
					}
				}
			});
		}
#pragma endregion
		//
		//#pragma region Fill crack
		//		{
		//			thrust::for_each(thrust::make_counting_iterator<size_t>(0), thrust::make_counting_iterator<size_t>(0) + depthMapInput.size(),
		//				[_result, hResolution, vResolution, xUnit, yUnit] __device__(size_t index) {
		//
		//				auto& p = _result[index];
		//				if (VECTOR3_VALID(p))
		//					return;
		//
		//				int yIndex = (int)(index / hResolution);
		//				int xIndex = (int)(index % hResolution);
		//
		//				_result[index] = Eigen::Vector3f(
		//					xUnit * (float)xIndex -((float)hResolution * xUnit * 0.5f),
		//					yUnit * (float)yIndex -((float)vResolution * yUnit * 0.5f),
		//					-20.f);
		//				return;
		//
		//				auto mean = Eigen::Vector3f(
		//					xUnit * (float)xIndex -((float)hResolution * xUnit * 0.5f),
		//					yUnit * (float)yIndex -((float)vResolution * yUnit * 0.5f),
		//					FLT_MAX);
		//				int validCount = 0;
		//				for (int y = -1; y < 2; y++)
		//				{
		//					for (int x = -1; x < 2; x++)
		//					{
		//						int ix = xIndex + x;
		//						int iy = yIndex + y;
		//
		//						if (ix < 0 || ix > hResolution)
		//							continue;
		//						if (iy < 0 || iy > vResolution)
		//							continue;
		//						if (x == 0 && y == 0)
		//							continue;
		//
		//						auto& np = _result[iy * hResolution + ix];
		//
		//						if (VECTOR3_VALID(np))
		//						{
		//							if (FLT_VALID(mean.z()))
		//							{
		//								mean += np;
		//							}
		//							else
		//							{
		//								mean.x() += np.x();
		//								mean.y() += np.y();
		//								mean.z() = np.z();
		//							}
		//							validCount++;
		//						}
		//					}
		//				}
		//
		//				if (validCount > 0)
		//				{
		//					//_result[index] = Eigen::Vector3f(
		//					//	xUnit * (float)xIndex -((float)hResolution * xUnit * 0.5f),
		//					//	yUnit * (float)yIndex -((float)vResolution * yUnit * 0.5f),
		//					//	-200.f);
		//					//return;
		//
		//					mean /= (float)validCount;
		//					_result[index] = mean;
		//				}
		//			});
		//		}
		//#pragma endregion

#pragma region Draw Result
		//{
		//	thrust::host_vector<Eigen::Vector3f> host_result(result.begin(), result.end());

		//	for (auto& p : host_result)
		//	{
		//		if (VECTOR3_VALID(p))
		//		{
		//			scene->Debug("Interpolation_Result")->AddPoint({ p.x(), p.y(), 0.0f }, glm::yellow);
		//			//scene->Debug("Interpolation_Result_Lines")->AddLine({ p.x(), p.y(), 0.0f }, { p.x(), p.y(), p.z() }, glm::yellow, glm::yellow);
		//		}

		//		if (FLT_VALID(p.x()))
		//		{
		//			float x = floorf(p.x() / xUnit) * xUnit + xUnit * 0.5f;
		//			float y = floorf(p.y() / yUnit) * yUnit + yUnit * 0.5f;
		//			float z = floorf(p.z() / xUnit) * xUnit + xUnit * 0.5f;
		//			//float z = (floorf(p.z() / xUnit) - 1) * xUnit + xUnit * 0.5f;

		//			scene->Debug("Interpolation_Result_Quantized")->AddPoint({ x, y, 0.0f }, glm::red);
		//			//scene->Debug("Interpolation_Result_Lines_Quantized_Lines")->AddLine({ x, y, 0.0f }, { x, y, z }, glm::red, glm::red);
		//		}
		//	}
		//}
#pragma endregion
	}

	void UpscaleDepthMap(Neon::Scene* scene, Neon::Mesh* mesh, size_t hResolution, size_t vResolution, float xUnit, float yUnit, float voxelSize, const thrust::device_vector<Eigen::Vector3f>& input, thrust::device_vector<Eigen::Vector3f>& result)
	{
		auto _input = thrust::raw_pointer_cast(input.data());
		auto _result = thrust::raw_pointer_cast(result.data());
		thrust::for_each(thrust::make_counting_iterator<size_t>(0),
			thrust::make_counting_iterator<size_t>(0) + vResolution * hResolution,
			[_input, _result, hResolution, vResolution, voxelSize]__device__(size_t index) {
			size_t yIndex = index / hResolution;
			size_t xIndex = index % hResolution;

			auto& inputPoint = _input[(yIndex / 2) * (hResolution / 2) + (xIndex / 2)];
			_result[(yIndex + 0) * hResolution + xIndex + 0] = Eigen::Vector3f(inputPoint.x(), inputPoint.y(), inputPoint.z());
			_result[(yIndex + 0) * hResolution + xIndex + 1] = Eigen::Vector3f(inputPoint.x() + voxelSize, inputPoint.y(), inputPoint.z());
			_result[(yIndex + 1) * hResolution + xIndex + 0] = Eigen::Vector3f(inputPoint.x(), inputPoint.y() + voxelSize, inputPoint.z());
			_result[(yIndex + 1) * hResolution + xIndex + 1] = Eigen::Vector3f(inputPoint.x() + voxelSize, inputPoint.y() + voxelSize, inputPoint.z());
		});
	}

	TSDF::TSDF()
	{
	}

	TSDF::TSDF(float voxelSize, const Eigen::Vector3f& inputMinPoint, const Eigen::Vector3f& inputMaxPoint)
		: voxelSize(voxelSize)
	{
		minPoint.x() = floorf(inputMinPoint.x() / voxelSize) * voxelSize;
		minPoint.y() = floorf(inputMinPoint.y() / voxelSize) * voxelSize;
		minPoint.z() = floorf(inputMinPoint.z() / voxelSize) * voxelSize;

		maxPoint.x() = ceilf(inputMaxPoint.x() / voxelSize) * voxelSize;
		maxPoint.y() = ceilf(inputMaxPoint.y() / voxelSize) * voxelSize;
		maxPoint.z() = ceilf(inputMaxPoint.z() / voxelSize) * voxelSize;

		centerPoint = (maxPoint + minPoint) * 0.5f;

		auto xLength = maxPoint.x() - minPoint.x();
		auto yLength = maxPoint.y() - minPoint.y();
		auto zLength = maxPoint.z() - minPoint.z();

		voxelCountX = (size_t)(xLength / voxelSize);
		voxelCountY = (size_t)(yLength / voxelSize);
		voxelCountZ = (size_t)(zLength / voxelSize);

		values = thrust::device_vector<float>(voxelCountZ * voxelCountY * voxelCountX, FLT_MAX);
		positions = thrust::device_vector<Eigen::Vector3f>(voxelCountZ * voxelCountY * voxelCountX, Eigen::Vector3f(0.0f, 0.0f, 0.0f));
		gridcells = thrust::device_vector<MarchingCubes::GRIDCELL>((voxelCountZ - 1) * (voxelCountY - 1) * (voxelCountX - 1));
		triangles = thrust::device_vector<MarchingCubes::TRIANGLE>(voxelCountZ * voxelCountY * voxelCountX * 4);

#pragma region Fill default values
		{
			nvtxRangePushA("@Arron/TSDF Fill");

			FillFunctor fillFunctor;
			fillFunctor.values = thrust::raw_pointer_cast(values.data());
			fillFunctor.positions = thrust::raw_pointer_cast(positions.data());
			fillFunctor.countX = voxelCountX;
			fillFunctor.countY = voxelCountY;
			fillFunctor.countZ = voxelCountZ;
			fillFunctor.minX = minPoint.x();
			fillFunctor.minY = minPoint.y();
			fillFunctor.minZ = minPoint.z();
			fillFunctor.voxelSize = voxelSize;

			thrust::for_each(thrust::make_counting_iterator<size_t>(0),
				thrust::make_counting_iterator<size_t>(values.size()), fillFunctor);

			nvtxRangePop();
		}
#pragma endregion

		printf("%d x %d x %d voxels\n", voxelCountX, voxelCountY, voxelCountZ);
		//printf("min : %f, %f, %f\n", minPoint.x(), minPoint.y(), minPoint.z());
		//printf("max : %f, %f, %f\n", maxPoint.x(), maxPoint.y(), maxPoint.z());
	}

	void TSDF::IntegrateMesh(Neon::Scene* scene, Neon::Mesh* mesh)
	{
		thrust::device_vector<Eigen::Vector3f> meshVertices;
		thrust::device_vector<GLuint> meshIndices;
	}

	void TSDF::IntegrateWrap(
		const std::vector<glm::vec3>& vertices,
		const Eigen::Matrix4f& transform,
		float width, float height,
		int columns, int rows)
	{
		nvtxRangePushA("@Arron/IntegrateWrap");

		auto inputVertices = FlipXInputArray(vertices, columns, rows);

		thrust::host_vector<Eigen::Vector3f> host_vertices;
		for (auto& v : vertices)
		{
			host_vertices.push_back(Eigen::Vector3f(v.x, v.y, v.z));
		}

		thrust::device_vector<Eigen::Vector3f> device_vertices(host_vertices.begin(), host_vertices.end());

		//cudaDeviceSynchronize();

		Integrate(device_vertices, transform, width, height, columns, rows);

		nvtxRangePop();
	}

	void TSDF::Integrate(
		const thrust::device_vector<Eigen::Vector3f>& vertices,
		const Eigen::Matrix4f& transform,
		float width, float height,
		int columns, int rows)
	{
		nvtxRangePushA("@Arron/Integrate");

		auto _values = thrust::raw_pointer_cast(values.data());
		auto _positions = thrust::raw_pointer_cast(positions.data());
		auto _inputPositions = thrust::raw_pointer_cast(vertices.data());

		IntegrateFunctor integrateFunctor;
		integrateFunctor.center = this->centerPoint;
		integrateFunctor.values = _values;
		integrateFunctor.positions = _positions;
		integrateFunctor.countX = voxelCountX;
		integrateFunctor.countY = voxelCountY;
		integrateFunctor.countZ = voxelCountZ;
		integrateFunctor.minX = this->minPoint.x();
		integrateFunctor.minY = this->minPoint.y();
		integrateFunctor.minZ = this->minPoint.z();
		integrateFunctor.voxelSize = voxelSize;
		integrateFunctor.inputPositions = _inputPositions;
		integrateFunctor.transform = transform;
		integrateFunctor.inverseTransform = transform.inverse();
		integrateFunctor.width = width;
		integrateFunctor.height = height;
		integrateFunctor.rows = rows;
		integrateFunctor.columns = columns;
		integrateFunctor.xUnit = width / columns;
		integrateFunctor.yUnit = height / rows;

		thrust::for_each(thrust::make_counting_iterator<size_t>(0),
			thrust::make_counting_iterator<size_t>(voxelCountX * voxelCountY * voxelCountZ),
			integrateFunctor);

		nvtxRangePop();
	}

	void TSDF::IntegrateDepthMapWrap(
		Neon::Scene* scene, Neon::Mesh* mesh,
		const Eigen::Matrix4f& transform,
		float width, float height,
		int columns, int rows)
	{
		//size_t hResolution = 256;
		//size_t vResolution = 480;
		//size_t uhResolution = 256 * 2;
		//size_t uvResolution = 480 * 2;
		//float xUnit = 0.1f;
		//float yUnit = 0.1f;
		//float voxelSize = 0.1f;

		//thrust::device_vector<Eigen::Vector3f> depthMap(hResolution * vResolution);

		//BuildDepthMap(scene, mesh, hResolution, vResolution, xUnit, yUnit, depthMap);

		//size_t voxelCountX = hResolution;
		//size_t voxelCountY = vResolution;
		//size_t voxelCountZ = hResolution;
		//size_t voxelCount = voxelCountX * voxelCountY * voxelCountZ;
		//thrust::device_vector<float> voxelValues(voxelCount);
		//auto _voxelValues = thrust::raw_pointer_cast(voxelValues.data());
		//auto _depthMap = thrust::raw_pointer_cast(depthMap.data());

		//thrust::for_each(thrust::make_counting_iterator<size_t>(0), thrust::make_counting_iterator<size_t>(voxelCount),
		//	[voxelSize, voxelCountX, voxelCountY, voxelCountZ, voxelCount, _voxelValues, _depthMap, uhResolution, uvResolution]__device__(size_t index) {

		//	auto z = index / (voxelCountX * voxelCountY);
		//	auto y = (index % (voxelCountX * voxelCountY)) / voxelCountX;
		//	auto x = (index % (voxelCountX * voxelCountY)) % voxelCountX;

		//	float xpos = (float)x * voxelSize - ((float)voxelCountX * voxelSize * 0.5f + voxelSize * 0.5f);
		//	float ypos = (float)y * voxelSize - ((float)voxelCountY * voxelSize * 0.5f + voxelSize * 0.5f);
		//	float zpos = (float)z * voxelSize - ((float)voxelCountZ * voxelSize * 0.5f + voxelSize * 0.5f);

		//	_voxelValues[index] = zpos - _depthMap[y * hResolution + x].z();
		//});

		//nvtxRangePushA("@Arron/IntegrateWrap");

		//thrust::host_vector<Eigen::Vector3f> host_vertices;
		//for (auto& v : vertices)
		//{
		//	host_vertices.push_back(Eigen::Vector3f(v.x, v.y, v.z));
		//}

		//thrust::device_vector<Eigen::Vector3f> device_vertices(host_vertices.begin(), host_vertices.end());

		////cudaDeviceSynchronize();

		//Integrate(device_vertices, transform, width, height, columns, rows);

		//nvtxRangePop();
	}

	void TSDF::IntegrateDepthMap(
		const thrust::device_vector<Eigen::Vector3f>& vertices,
		const Eigen::Matrix4f& transform,
		float width, float height,
		int columns, int rows)
	{
		nvtxRangePushA("@Arron/Integrate");

		auto _values = thrust::raw_pointer_cast(values.data());
		auto _positions = thrust::raw_pointer_cast(positions.data());
		auto _inputPositions = thrust::raw_pointer_cast(vertices.data());

		IntegrateFunctor integrateFunctor;
		integrateFunctor.center = this->centerPoint;
		integrateFunctor.values = _values;
		integrateFunctor.positions = _positions;
		integrateFunctor.countX = voxelCountX;
		integrateFunctor.countY = voxelCountY;
		integrateFunctor.countZ = voxelCountZ;
		integrateFunctor.minX = this->minPoint.x();
		integrateFunctor.minY = this->minPoint.y();
		integrateFunctor.minZ = this->minPoint.z();
		integrateFunctor.voxelSize = voxelSize;
		integrateFunctor.inputPositions = _inputPositions;
		integrateFunctor.transform = transform;
		integrateFunctor.inverseTransform = transform.inverse();
		integrateFunctor.width = width;
		integrateFunctor.height = height;
		integrateFunctor.rows = rows;
		integrateFunctor.columns = columns;
		integrateFunctor.xUnit = width / columns;
		integrateFunctor.yUnit = height / rows;

		thrust::for_each(thrust::make_counting_iterator<size_t>(0),
			thrust::make_counting_iterator<size_t>(voxelCountX * voxelCountY * voxelCountZ),
			integrateFunctor);

		nvtxRangePop();
	}

	void TSDF::UpdateValues()
	{
		nvtxRangePushA("@Arron/UpdateValues");

		auto _values = thrust::raw_pointer_cast(values.data());
		auto _positions = thrust::raw_pointer_cast(positions.data());

		UpdateVoxelsFunctor updateVoxelsFunctor;
		updateVoxelsFunctor.center = this->centerPoint;
		updateVoxelsFunctor.values = _values;
		updateVoxelsFunctor.positions = _positions;
		updateVoxelsFunctor.countX = voxelCountX;
		updateVoxelsFunctor.countY = voxelCountY;
		updateVoxelsFunctor.countZ = voxelCountZ;
		updateVoxelsFunctor.minX = this->minPoint.x();
		updateVoxelsFunctor.minY = this->minPoint.y();
		updateVoxelsFunctor.minZ = this->minPoint.z();
		updateVoxelsFunctor.voxelSize = voxelSize;

		thrust::for_each(thrust::make_counting_iterator<size_t>(0),
			thrust::make_counting_iterator<size_t>(voxelCountX * voxelCountY * voxelCountZ),
			updateVoxelsFunctor);

		nvtxRangePop();
	}

	void TSDF::BuildGridCells(float isoValue)
	{
		nvtxRangePushA("@Arron/BuildGridCells");

		auto _values = thrust::raw_pointer_cast(values.data());
		//auto _positions = thrust::raw_pointer_cast(positions.data());
		auto _gridcells = thrust::raw_pointer_cast(gridcells.data());
		auto _triangles = thrust::raw_pointer_cast(triangles.data());

		BuildGridFunctor buildGridFunctor;
		buildGridFunctor.center = this->centerPoint;
		buildGridFunctor.values = _values;
		buildGridFunctor.gridcells = _gridcells;
		//buildGridFunctor.positions = _positions;
		buildGridFunctor.triangles = _triangles;
		buildGridFunctor.countX = voxelCountX;
		buildGridFunctor.countY = voxelCountY;
		buildGridFunctor.countZ = voxelCountZ;
		buildGridFunctor.minX = this->minPoint.x();
		buildGridFunctor.minY = this->minPoint.y();
		buildGridFunctor.minZ = this->minPoint.z();
		buildGridFunctor.voxelSize = voxelSize;
		buildGridFunctor.isoValue = isoValue;

		thrust::for_each(thrust::make_counting_iterator<size_t>(0),
			thrust::make_counting_iterator<size_t>((voxelCountX - 1) * (voxelCountY - 1) * (voxelCountZ - 1)),
			buildGridFunctor);

		nvtxRangePop();
	}

	void TSDF::TestValues(Neon::Scene* scene)
	{
		thrust::host_vector<float> host_values(values.begin(), values.end());

		int cnt = 0;

		for (size_t z = 0; z < voxelCountZ; z++)
		{
			char buffer[16]{ 0 };
			itoa(z, buffer, 10);

			auto debug = scene->Debug(buffer);

			for (size_t y = 0; y < voxelCountY; y++)
			{
				for (size_t x = 0; x < voxelCountX; x++)
				{
					Eigen::Vector3f point(
						minPoint.x() + voxelSize * x,
						minPoint.y() + voxelSize * y,
						minPoint.z() + voxelSize * z);

					auto value = host_values[z * voxelCountY * voxelCountX + y * voxelCountX + x];

					if (cnt == 100)
					{
						if (value != FLT_MAX)
						{
							//debug->AddBox({ point.x(), point.y(), point.z() }, voxelSize, voxelSize, voxelSize, glm::vec4((10.0f - value) / 10.0f, value / 10.0f, 1.0f, (10.0f - value) / 10.0f));
							debug->AddBox({ point.x(), point.y(), point.z() }, voxelSize, voxelSize, voxelSize, glm::vec4((10.0f - value) / 10.0f, value / 10.0f, 1.0f, (10.0f - value) / 10.0f));
						}
						cnt = 0;
					}

					cnt++;
				}
			}
		}
	}

	void TSDF::TestTriangles(Neon::Scene* scene)
	{
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
			}
		}
	}

	void TSDF::TestInput(Neon::Scene* scene, const Eigen::Matrix4f& transform, Neon::Mesh* mesh, const glm::vec4& color)
	{
		size_t columns = 256;
		size_t rows = 480;

		//float xUnit = width / columns;
		//float yUnit = height / rows;

		float xUnit = 0.1f;
		float yUnit = 0.1f / 3;

		float width = xUnit * columns;
		float height = yUnit * rows;

		//float minX = mesh->GetAABB().GetMinPoint().x;
		//float minY = mesh->GetAABB().GetMinPoint().y;

		//float width = mesh->GetAABB().GetXLength();
		//float height = mesh->GetAABB().GetYLength();

		//float cx = mesh->GetAABB().GetCenter().x;
		//float cy = mesh->GetAABB().GetCenter().y;

		scene->Debug("BoxLine")->AddLine({ -width * 0.5f, -height * 0.5f, 0.0f }, { width * 0.5f, -height * 0.5f, 0.0f });
		scene->Debug("BoxLine")->AddLine({ -width * 0.5f, -height * 0.5f, 0.0f }, { -width * 0.5f,  height * 0.5f, 0.0f });
		scene->Debug("BoxLine")->AddLine({ width * 0.5f,  height * 0.5f, 0.0f }, { -width * 0.5f,  height * 0.5f, 0.0f });
		scene->Debug("BoxLine")->AddLine({ width * 0.5f,  height * 0.5f, 0.0f }, { width * 0.5f, -height * 0.5f, 0.0f });
		scene->Debug("BoxLine")->AddLine({ -width * 0.5f,  0.0f, 0.0f }, { width * 0.5f, 0.0f, 0.0f });
		scene->Debug("BoxLine")->AddLine({ 0.0f, height * 0.5f, 0.0f }, { 0.0f, -height * 0.5f, 0.0f });


		auto inputPositions = FlipXInputArray(mesh->GetVertexBuffer()->GetElements(), columns, rows);

		auto im = transform.inverse();

		for (size_t row = 0; row < rows; row++)
		{
			for (size_t column = 0; column < columns; column++)
			{
				auto index = row * columns + column;

				auto& v = inputPositions[index];

				if (FLT_VALID(v.x) && FLT_VALID(v.y) && FLT_VALID(v.z))
				{
					scene->Debug("input points")->AddPoint({ v.x, v.y, 0.0f });

					////////auto gridx = column * xUnit * 2 - (width * 0.5f);
					////////auto gridx = column * xUnit * 2 - width;
					//////auto gridx = (float)column * xUnit - ((float)columns * xUnit * 0.5f);
					//////auto gridxRight = ((float)column + 1) * xUnit - ((float)columns * xUnit * 0.5f);
					//////auto gridy = -(float)rows * 0.5f * yUnit + (float)row * yUnit;

					////////auto gridx = column * xUnit;
					////////auto gridxRight = (column + 1) * xUnit;
					////////auto gridy = row * yUnit;

					////////if (v.x < gridx || gridxRight < v.x)
					//////{
					//////	//printf("gridx : %f, gridRight : %f, v.x : %f\n", gridx, gridxRight, v.x);

					//////	scene->Debug("input points")->AddPoint({ gridx, gridy, 1.0f }, glm::red);
					//////}
				}
			}
		}

		//for (auto& v : inputPositions)
		//{
		//	scene->Debug("input points")->AddPoint({v.x, v.y, 0.0f});
		//}

		//for (size_t z = 0; z < voxelCountZ; z++)
		//{
		//	for (size_t y = 0; y < voxelCountY; y++)
		//	{
		//		for (size_t x = 0; x < voxelCountX; x++)
		//		{
		//			Eigen::Vector4f point(
		//				minPoint.x() + voxelSize * x + 0.5f * voxelSize,
		//				minPoint.y() + voxelSize * y + 0.5f * voxelSize,
		//				minPoint.z() + voxelSize * z + 0.5f * voxelSize,
		//				1.0f);

		//			auto ip = im * point;

		//			scene->Debug("voxel points")->AddPoint({ ip.x(), ip.y(), ip.z() }, color);
		//		}
		//	}
		//}
	}

	void TSDF::ShowInversedVoxels(Neon::Scene* scene, const Eigen::Matrix4f& transform, Neon::Mesh* mesh)
	{
		float width = mesh->GetAABB().GetXLength();
		float height = mesh->GetAABB().GetYLength();

		float columns = 256;
		float rows = 480;

		float xUnit = width / columns;
		float yUnit = height / rows;

		auto inputPositions = FlipXInputArray(mesh->GetVertexBuffer()->GetElements(), columns, rows);

		auto im = transform.inverse();
		//auto im = transform;

		auto org = im * Eigen::Vector4f(0.0f, 0.0f, 0.0f, 1.0f);
		scene->Debug("orgin")->AddBox({ org.x(), org.y(), org.z() }, 1, 1, 1, glm::red);

		for (size_t z = 0; z < voxelCountZ; z++)
		{
			for (size_t y = 0; y < voxelCountY; y++)
			{
				for (size_t x = 0; x < voxelCountX; x++)
				{
					Eigen::Vector4f point(
						minPoint.x() + voxelSize * x + 0.5f * voxelSize,
						minPoint.y() + voxelSize * y + 0.5f * voxelSize,
						minPoint.z() + voxelSize * z + 0.5f * voxelSize,
						1.0f);

					//scene->Debug("inputPoint Line")->AddLine({ point.x(), point.y(), point.z()}, {point.x(), point.y(), 0.0f}, glm::red, glm::blue);

					Eigen::Vector4f ip = im * point;
					Eigen::Vector4f ip0 = im * Eigen::Vector4f(point.x(), point.y(), 0.0f, 1.0f);

					float xPosition = ip.x() + width * 0.5f;
					float yPosition = ip.y() + height * 0.5f;
					float distance = ip.z();

					//scene->Debug("inputPoint")->AddPoint({ ip.x(), ip.y(), ip.z() }, glm::red);
					//scene->Debug("inputPoint Line")->AddLine({ ip.x(), ip.y(), ip.z() }, { ip.x(), ip.y(), 0.0f });

					if ((0 < xPosition && xPosition < width) &&
						(0 < yPosition && yPosition < height))
					{
						int xIndex = ((int)(xPosition / xUnit) / 2) * 2;
						int yIndex = ((int)(yPosition / yUnit) / 3) * 3;
						//int xIndex = (int)floorf(xPosition / xUnit);
						//int yIndex = (int)floorf(yPosition / yUnit);

						auto inputPosition = inputPositions[yIndex * columns + xIndex];

						if (FLT_VALID(inputPosition.x) && FLT_VALID(inputPosition.y) && FLT_VALID(inputPosition.z))
						{
							//printf("%f %f %f\n", inputPosition.x, inputPosition.y, inputPosition.z);

							scene->Debug("inputPoint Line")->AddLine({ inputPosition.x, inputPosition.y, inputPosition.z }, { ip.x(), ip.y(), ip.z() }, glm::red, glm::blue);
						}

						//if (FLT_VALID(inputPosition.x()) && FLT_VALID(inputPosition.y()) && FLT_VALID(inputPosition.z()))
						//{
						//	//printf("[%d, %d] inputPosition : %f, %f, %f\n", xIndex, yIndex, inputPosition.x(), inputPosition.y(), inputPosition.z());
						//	//values[index] = distance - inputPosition.z();

						//	//printf("%f\n", (inputPosition - Eigen::Vector3f(ip.x(), ip.y(), ip.z())).norm());
						//}
						//else
						//{
						//	//values[index] = FLT_MAX;
						//	//	//values[index] = 0.0f;
						//}
					}


					//scene->Debug("original")->AddPoint({ point.x(), point.y(), point.z() }, glm::green);

					//scene->Debug("inversed")->AddPoint({ ip.x(), ip.y(), ip.z() }, glm::yellow);
				}
			}
		}
	}

	bool TSDF::ShowInversedVoxelsSingle(Neon::Scene* scene, const Eigen::Matrix4f& transform, Neon::Mesh* mesh, int singleIndex)
	{
		float width = mesh->GetAABB().GetXLength();
		float height = mesh->GetAABB().GetYLength();

		float columns = 256;
		float rows = 480;

		float xUnit = width / columns;
		float yUnit = height / rows;

		auto meshCenter = mesh->GetAABB().center;

		auto inputPositions = FlipXInputArray(mesh->GetVertexBuffer()->GetElements(), columns, rows);

		Eigen::Matrix4f translationMatrix = Eigen::Matrix4f::Identity();
		translationMatrix.block<3, 1>(0, 3).x() = meshCenter.x;
		translationMatrix.block<3, 1>(0, 3).y() = meshCenter.y;
		translationMatrix.block<3, 1>(0, 3).z() = meshCenter.z;

		auto tm = transform * translationMatrix;

		auto im = tm.inverse();

		int z = singleIndex / (voxelCountY * voxelCountX);
		int y = (singleIndex % (voxelCountY * voxelCountX)) / voxelCountX;
		int x = (singleIndex % (voxelCountY * voxelCountX)) % voxelCountX;

		//printf("[index : %6d] x: %d, y: %d, z: %d\n", singleIndex, x, y, z);

		{
			Eigen::Vector4f point(
				minPoint.x() + voxelSize * x + 0.5f * voxelSize,
				minPoint.y() + voxelSize * y + 0.5f * voxelSize,
				minPoint.z() + voxelSize * z + 0.5f * voxelSize,
				1.0f);

			//scene->Debug("inputPoint Line")->AddLine({ point.x(), point.y(), point.z()}, {point.x(), point.y(), 0.0f}, glm::red, glm::blue);

			Eigen::Vector4f ip = im * point;

			float xPosition = ip.x() + width * 0.5f;
			float yPosition = ip.y() + height * 0.5f;
			float distance = ip.z();

			if ((0 < xPosition && xPosition < width) &&
				(0 < yPosition && yPosition < height))
			{
				int xIndex = ((int)(xPosition / xUnit) / 2) * 2;
				int yIndex = ((int)(yPosition / yUnit) / 3) * 3;

				auto inputPosition = inputPositions[yIndex * columns + xIndex];

				//printf("\n");
				//printf("[index    : %6d]\nx: %d, y: %d, z: %d\n", singleIndex, x, y, z);
				//printf("ip        : %4.6f, %4.6f, %4.6f\n", ip.x(), ip.y(), ip.z());
				//printf("xPosition : %4.6f,\tyPoisition : %4.6f,\tdistance : %4.6f\n", xPosition, yPosition, distance);
				//printf("xUnit     : %4.6f,\tyUnit : %4.6f\n", xUnit, yUnit);
				//printf("xIndex    : %6d,\tyIndex : %6d\n", xIndex, yIndex);
				//printf("x         : %4.6f\n", -width * 0.5f + xIndex * xUnit);
				//printf("inputPosition x : %f\n", inputPosition.x);
				//int tempIndex = 0;
				//float beginX = inputPositions[yIndex * columns + tempIndex].x;
				//while (beginX > width)
				//{
				//	beginX = inputPositions[yIndex * columns + tempIndex++].x;
				//}
				//printf("row beginX: %4.6f\n", beginX);
				//printf("beginX index :%d\n", tempIndex);
				//return true;

				if (FLT_VALID(inputPosition.x) && FLT_VALID(inputPosition.y) && FLT_VALID(inputPosition.z))
				{
					printf("\n");
					printf("[index    : %6d]\nx: %d, y: %d, z: %d\n", singleIndex, x, y, z);
					printf("ip        : %4.6f, %4.6f, %4.6f\n", ip.x(), ip.y(), ip.z());
					printf("xPosition : %4.6f,\tyPoisition : %4.6f,\tdistance : %4.6f\n", xPosition, yPosition, distance);
					printf("xUnit     : %4.6f,\tyUnit : %4.6f\n", xUnit, yUnit);
					printf("xIndex    : %6d,\tyIndex : %6d\n", xIndex, yIndex);
					printf("x         : %4.6f\n", -width * 0.5f + xIndex * xUnit);

					printf("%f %f %f\n", inputPosition.x, inputPosition.y, inputPosition.z);

					scene->Debug("inputPoint")->AddPoint({ point.x(), point.y(), point.z() });
					scene->Debug("inputPoint")->AddPoint({ ip.x(), ip.y(), ip.z() }, glm::red);

					scene->Debug("inputPoint Line")->AddLine({ inputPosition.x, inputPosition.y, inputPosition.z }, { ip.x(), ip.y(), ip.z() }, glm::red, glm::blue);

					scene->Debug("inputPoint Line")->AddLine({ ip.x(), ip.y(), ip.z() }, { ip.x(), ip.y(), 0.0f });

					return true;
				}
			}
			//scene->Debug("original")->AddPoint({ point.x(), point.y(), point.z() }, glm::green);
			//scene->Debug("inversed")->AddPoint({ ip.x(), ip.y(), ip.z() }, glm::yellow);
		}

		return false;
	}
}
