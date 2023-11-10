#include <Neon/CUDA/CUDAMesh.h>

#include <Neon/NeonCommon.h>

namespace NeonCUDA
{
	__global__ void SubdivideKernel(const glm::vec3* inVertices, const unsigned int* inIndices, glm::vec3* outVertices, unsigned int* outIndices, int numTriangles) {
		int tid = threadIdx.x + blockIdx.x * blockDim.x;
		if (tid < numTriangles) {
			int baseIndex = tid * 3;
			int newVertexIndex = tid * 6;
			int newIndex = tid * 12;

			// Get the vertices of the current triangle
			glm::vec3 v0 = inVertices[inIndices[baseIndex]];
			glm::vec3 v1 = inVertices[inIndices[baseIndex + 1]];
			glm::vec3 v2 = inVertices[inIndices[baseIndex + 2]];

			// Calculate midpoints of the edges
			glm::vec3 mid01 = 0.5f * (v0 + v1);
			glm::vec3 mid12 = 0.5f * (v1 + v2);
			glm::vec3 mid20 = 0.5f * (v2 + v0);

			// Add the new vertices to the output array
			outVertices[newVertexIndex + 0] = v0;
			outVertices[newVertexIndex + 1] = v1;
			outVertices[newVertexIndex + 2] = v2;
			outVertices[newVertexIndex + 3] = mid01;
			outVertices[newVertexIndex + 4] = mid12;
			outVertices[newVertexIndex + 5] = mid20;

			// Update the indices for the new triangles
			outIndices[newIndex] = newVertexIndex;
			outIndices[newIndex + 1] = newVertexIndex + 3;
			outIndices[newIndex + 2] = newVertexIndex + 5;

			outIndices[newIndex + 3] = newVertexIndex + 3;
			outIndices[newIndex + 4] = newVertexIndex + 1;
			outIndices[newIndex + 5] = newVertexIndex + 4;

			outIndices[newIndex + 6] = newVertexIndex + 2;
			outIndices[newIndex + 7] = newVertexIndex + 5;
			outIndices[newIndex + 8] = newVertexIndex + 4;

			outIndices[newIndex + 9] = newVertexIndex + 3;
			outIndices[newIndex + 10] = newVertexIndex + 4;
			outIndices[newIndex + 11] = newVertexIndex + 5;
		}
	}

	CUDAMesh::CUDAMesh()
	{
	}

	CUDAMesh::~CUDAMesh()
	{
	}

	size_t CUDAMesh::AddVertex(const glm::vec3& v)
	{
		host_vertices.push_back(v);
		return host_vertices.size() - 1;
	}

	size_t CUDAMesh::AddIndex(unsigned int i)
	{
		host_indices.push_back(i);
		return host_indices.size() - 1;
	}

	size_t CUDAMesh::AddTriangle(unsigned int i0, unsigned int i1, unsigned int i2)
	{
		host_indices.push_back(i0);
		host_indices.push_back(i1);
		host_indices.push_back(i2);

		return (host_indices.size() - 1) / 3;
	}

	void CUDAMesh::Upload()
	{
		auto t = Neon::Time("Upload()");

		device_vertices = host_vertices;
		device_indices = host_indices;
	}

	void CUDAMesh::Download()
	{
		auto t = Neon::Time("Download()");

		host_vertices = device_vertices;
		host_indices = device_indices;
	}

	void CUDAMesh::Subdivide()
	{
		auto t = Neon::Time("Subdivide()");

		int numTriangles = (int)host_indices.size() / 3;

		// Create arrays for the subdivided mesh data on the GPU
		thrust::device_vector<glm::vec3> subdividedVertices(device_vertices.size() * 2);
		thrust::device_vector<unsigned int> subdividedIndices(numTriangles * 12);

		// Launch the CUDA kernel for subdivision
		int threadsPerBlock = 256;
		int numBlocks = (numTriangles + threadsPerBlock - 1) / threadsPerBlock;

		SubdivideKernel<<<numBlocks, threadsPerBlock>>>(thrust::raw_pointer_cast(device_vertices.data()),
			thrust::raw_pointer_cast(device_indices.data()),
			thrust::raw_pointer_cast(subdividedVertices.data()),
			thrust::raw_pointer_cast(subdividedIndices.data()),
			numTriangles);

		cudaDeviceSynchronize();

		// Transfer the subdivided data back to the CPU
		//thrust::host_vector<glm::vec3> hostSubdividedVertices(subdividedVertices);
		//thrust::host_vector<unsigned int> hostSubdividedIndices(subdividedIndices);

		//device_vertices = thrust::device_vector<glm::vec3>(subdividedVertices);
		//device_indices = thrust::device_vector<unsigned int>(subdividedIndices);

		host_vertices.clear();
		host_indices.clear();

		host_vertices = subdividedVertices;
		host_indices = subdividedIndices;
	}
}
