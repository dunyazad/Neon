#include "cuOctree.h"

#define MAX_POINTS_PER_NODE 8
#define INITIAL_MAX_NODES 1000000

#include <iostream>

struct f3 {
    float x, y, z;
    __host__ __device__ f3() : x(0), y(0), z(0) {}
    __host__ __device__ f3(float x, float y, float z) : x(x), y(y), z(z) {}
    __host__ __device__ f3 operator+(const f3& b) const {
        return f3(x + b.x, y + b.y, z + b.z);
    }
    __host__ __device__ f3 operator*(float s) const {
        return f3(x * s, y * s, z * s);
    }
    __host__ __device__ f3 operator/(float s) const {
        return f3(x / s, y / s, z / s);
    }
};

struct OctreeNode {
    f3 min_bound;  // Minimum bound of the node
    f3 max_bound;  // Maximum bound of the node
    int point_count;   // Number of points in the node
    f3 points[MAX_POINTS_PER_NODE];  // Points in the node
    int children[8]; // Indices of child nodes

    __device__ __host__ OctreeNode() : point_count(0) {
        for (int i = 0; i < 8; ++i) {
            children[i] = -1;
        }
    }

    __device__ __host__ OctreeNode(f3 min_b, f3 max_b)
        : min_bound(min_b), max_bound(max_b), point_count(0) {
        for (int i = 0; i < 8; ++i) {
            children[i] = -1;
        }
    }
};

__device__ int getChildIndex(const f3& point, const f3& mid_point) {
    int index = 0;
    if (point.x > mid_point.x) index |= 1;
    if (point.y > mid_point.y) index |= 2;
    if (point.z > mid_point.z) index |= 4;
    return index;
}

//__global__ void insertPoints(OctreeNode* nodes, int* node_counter, f3* points, int num_points, int max_nodes) {
//    int idx = blockIdx.x * blockDim.x + threadIdx.x;
//    if (idx >= num_points) return;
//
//    f3 point = points[idx];
//    int node_index = 0; // Start at root
//    OctreeNode* node = &nodes[node_index];
//
//    while (true) {
//        int point_idx = atomicAdd(&node->point_count, 1);
//
//        if (point_idx < MAX_POINTS_PER_NODE) {
//            node->points[point_idx] = point;
//            return;
//        }
//        else {
//            atomicSub(&node->point_count, 1); // Undo the increment
//
//            f3 mid_point = (node->min_bound + node->max_bound) * 0.5f;
//            int child_idx = getChildIndex(point, mid_point);
//
//            if (node->children[child_idx] == -1) {
//                int new_node_idx = atomicAdd(node_counter, 1);
//                if (new_node_idx >= max_nodes) {
//                    printf("[%d / %d] Error: Exceeded maximum number of nodes\n", new_node_idx, max_nodes);
//                    return;
//                }
// 
//                //printf("[%d / %d]\n", new_node_idx, max_nodes);
//
//                f3 new_min_bound = node->min_bound;
//                f3 new_max_bound = node->max_bound;
//                if (child_idx & 1) new_min_bound.x = mid_point.x; else new_max_bound.x = mid_point.x;
//                if (child_idx & 2) new_min_bound.y = mid_point.y; else new_max_bound.y = mid_point.y;
//                if (child_idx & 4) new_min_bound.z = mid_point.z; else new_max_bound.z = mid_point.z;
//
//                nodes[new_node_idx] = OctreeNode(new_min_bound, new_max_bound);
//                node->children[child_idx] = new_node_idx;
//            }
//
//            node_index = node->children[child_idx];
//            node = &nodes[node_index];
//        }
//    }
//}



__global__ void classifyPoints(OctreeNode* nodes, int* node_counter, f3* points, int num_points, int* point_indices, int max_nodes) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_points) return;

    f3 point = points[idx];
    int node_index = 0; // Start at root
    OctreeNode* node = &nodes[node_index];

    while (true) {
        f3 mid_point = (node->min_bound + node->max_bound) * 0.5f;
        int child_idx = getChildIndex(point, mid_point);

        if (node->children[child_idx] == -1) {
            int new_node_idx = atomicAdd(node_counter, 1);
            if (new_node_idx >= max_nodes) {
                printf("Error: Exceeded maximum number of nodes\n");
                return;
            }

            f3 new_min_bound = node->min_bound;
            f3 new_max_bound = node->max_bound;
            if (child_idx & 1) new_min_bound.x = mid_point.x; else new_max_bound.x = mid_point.x;
            if (child_idx & 2) new_min_bound.y = mid_point.y; else new_max_bound.y = mid_point.y;
            if (child_idx & 4) new_min_bound.z = mid_point.z; else new_max_bound.z = mid_point.z;

            nodes[new_node_idx] = OctreeNode(new_min_bound, new_max_bound);
            node->children[child_idx] = new_node_idx;
        }

        node_index = node->children[child_idx];
        node = &nodes[node_index];

        if (node->point_count < MAX_POINTS_PER_NODE) {
            point_indices[idx] = node_index;
            return;
        }
    }
}

__global__ void insertPoints(OctreeNode* nodes, f3* points, int* point_indices, int num_points) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_points) return;

    int node_index = point_indices[idx];
    OctreeNode* node = &nodes[node_index];

    int point_idx = atomicAdd(&node->point_count, 1);
    if (point_idx < MAX_POINTS_PER_NODE) {
        node->points[point_idx] = points[idx];
    }
}

void RunOctreeExample(Neon::Scene* scene)
{
    const int num_points = 8000000;
    f3* d_points;
    f3* h_points = new f3[num_points];

    // Initialize points
    for (int i = 0; i < num_points; ++i) {
        h_points[i] = f3(2000.0f * (float)rand() / RAND_MAX, 2000.0f * (float)rand() / RAND_MAX, 2000.0f * (float)rand() / RAND_MAX);

        scene->Debug("Points")->AddPoint({ h_points[i].x, h_points[i].y, h_points[i].z });
    }

    cudaMalloc(&d_points, num_points * sizeof(f3));
    cudaMemcpy(d_points, h_points, num_points * sizeof(f3), cudaMemcpyHostToDevice);

    OctreeNode* d_nodes;
    int max_nodes = INITIAL_MAX_NODES;
    cudaMalloc(&d_nodes, max_nodes * sizeof(OctreeNode));
    int* d_node_counter = nullptr;
    cudaMalloc(&d_node_counter, sizeof(int));
    int h_node_counter = 1;
    cudaMemcpy(d_node_counter, &h_node_counter, sizeof(int), cudaMemcpyHostToDevice); // Start counter at 1 (root is index 0)

    OctreeNode* h_root = new OctreeNode(f3(0, 0, 0), f3(1, 1, 1));
    cudaMemcpy(&d_nodes[0], h_root, sizeof(OctreeNode), cudaMemcpyHostToDevice);

    int* d_point_indices;
    cudaMalloc(&d_point_indices, num_points * sizeof(int));

    cudaDeviceSynchronize();

    nvtxRangePushA("Insert");

    int threads_per_block = 256;
    int blocks = (num_points + threads_per_block - 1) / threads_per_block;
    classifyPoints << <blocks, threads_per_block >> > (d_nodes, d_node_counter, d_points, num_points, d_point_indices, max_nodes);
    cudaDeviceSynchronize();

    insertPoints << <blocks, threads_per_block >> > (d_nodes, d_points, d_point_indices, num_points);
    cudaDeviceSynchronize();

    cudaDeviceSynchronize();

    nvtxRangePop();

    cudaMemcpy(&h_node_counter, d_node_counter, sizeof(int), cudaMemcpyDeviceToHost);

    if (h_node_counter >= max_nodes) {
        std::cerr << "Exceeded maximum number of nodes during insertion." << std::endl;
    }
    else {
        std::cout << "Insertion completed successfully with " << h_node_counter << " nodes." << std::endl;
    }

    // Cleanup
    cudaFree(d_points);
    cudaFree(d_nodes);
    cudaFree(d_node_counter);
    cudaFree(d_point_indices);
    delete[] h_points;
    delete h_root;
}