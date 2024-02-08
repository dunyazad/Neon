#include <glad/gl.h>

#include <iostream>
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>
#include <GL/gl.h>
#include <GLFW/glfw3.h>

const int width = 800;
const int height = 600;
GLFWwindow* window;

cudaGraphicsResource_t cuda_vbo_resource;

// CUDA kernel
__global__ void cudaKernel(float4* buffer, int width, int height, float time) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) {
        int index = y * width + x;
        float frequency = 0.1f;
        float amplitude = 1.0f;

        float value = sinf(frequency * (x + time)) * amplitude;
        buffer[index] = make_float4(value, value, value, 1.0f);
    }
}

// OpenGL render function
void render() {
    // Run CUDA kernel to update the buffer
    float4* d_buffer;
    cudaGraphicsMapResources(1, &cuda_vbo_resource, 0);
    size_t num_bytes;
    cudaGraphicsResourceGetMappedPointer((void**)&d_buffer, &num_bytes, cuda_vbo_resource);

    dim3 block(16, 16);
    dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y);

    cudaKernel << <grid, block >> > (d_buffer, width, height, glfwGetTime());

    cudaGraphicsUnmapResources(1, &cuda_vbo_resource, 0);

    // Render the buffer using OpenGL
    glDrawPixels(width, height, GL_RGBA, GL_FLOAT, 0);

    glfwSwapBuffers(window);
    glfwPollEvents();
}
//
//int main() {
//    // Initialize GLFW
//    if (!glfwInit()) {
//        std::cerr << "Failed to initialize GLFW" << std::endl;
//        return -1;
//    }
//
//    // Create a windowed mode window and its OpenGL context
//    window = glfwCreateWindow(width, height, "CUDA-OpenGL Interoperability Example", NULL, NULL);
//    if (!window) {
//        glfwTerminate();
//        return -1;
//    }
//
//    // Make the window's context current
//    glfwMakeContextCurrent(window);
//
//    // Set up OpenGL parameters
//    glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
//    glOrtho(0, width, 0, height, -1, 1);
//
//    // Create a CUDA-OpenGL buffer
//    GLuint vbo;
//    glGenBuffers(1, &vbo);
//    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, vbo);
//    glBufferData(GL_PIXEL_UNPACK_BUFFER, width * height * sizeof(float4), 0, GL_DYNAMIC_DRAW);
//
//    // Register the buffer with CUDA
//    cudaGraphicsGLRegisterBuffer(&cuda_vbo_resource, vbo, cudaGraphicsMapFlagsWriteDiscard);
//
//    // Main loop
//    while (!glfwWindowShouldClose(window)) {
//        render();
//    }
//
//    // Clean up resources
//    cudaGraphicsUnregisterResource(cuda_vbo_resource);
//    glBindBuffer(1, vbo);
//    glDeleteBuffers(1, &vbo);
//
//    // Terminate GLFW
//    glfwTerminate();
//
//    return 0;
//}
