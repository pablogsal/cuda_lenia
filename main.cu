#include <GL/glew.h>
#include <GL/freeglut.h>
#include <cstdlib>
#include <cuda_gl_interop.h>
#include <cuda_runtime.h>
#include <fstream>
#include <iostream>
#include <vector>

const int WIDTH = 1000;
const int HEIGHT = 1000;
const int NUM_THREADS_X = 16;
const int NUM_THREADS_Y = 16;
const int NUM_BLOCKS_X = WIDTH / NUM_THREADS_X;
const int NUM_BLOCKS_Y = HEIGHT / NUM_THREADS_Y;
const int NUM_PARTICLES = 400;
const int SIZE = 20;

// Global Constants for Parameters
const float mu_k = 4.0f;
const float sigma_k = 1.0f;
const float w_k = 0.022f;
const float mu_g = 0.6f;
const float sigma_g = 0.15f;
const float c_rep = 1.0f;
const float dt = 0.1f;

int currentStep = 0;

float* d_world = nullptr;
float3* d_particlePositions = nullptr;
cudaGraphicsResource* cudaGLResource;
float3 particlePositions[NUM_PARTICLES];

__device__ inline float
fast_exp(float x)
{
    float t = 1.0f + x / 32.0f;
    t *= t;
    t *= t;
    t *= t;
    t *= t;
    t *= t;  // t **= 32
    return t;
}

__device__ inline float
peak_f(const float x, const float mu, const float sigma, const float w = 1.0f)
{
    float t = (x - mu) / sigma;
    float y = w / fast_exp(t * t);
    return y;
}

__device__ inline float
peak_fd(const float x, const float mu, const float sigma, const float w = 1.0f)
{
    float t = (x - mu) / sigma;
    float y = w / fast_exp(t * t);
    return -2.0f * t * y / sigma;
}

__device__ inline float
repulsion_f(float x, float c_rep)
{
    float t = std::max(1.0f - x, 0.0f);
    return 0.5f * c_rep * t * t;
}

__device__ inline float
repulsion_fd(float x, float c_rep)
{
    float t = std::max(1.0f - x, 0.0f);
    return -c_rep * t;
}

// CUDA kernel for solving the heat equation with shared memory
__global__ void
energySolver(float* d_world, float3* d_particles, int width, int height)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    float repulsion = 0.0f;
    float attraction = 0.0f;

    // Calculate total energy for this cell from all particles
    for (int k = 0; k < NUM_PARTICLES; ++k) {
        float3 p = d_particles[k];
        float x0 = (static_cast<float>(i) - WIDTH / 2.0f) / WIDTH * SIZE;
        float y0 = (static_cast<float>(j) - HEIGHT / 2.0f) / HEIGHT * SIZE;
        float x = x0 - p.x;
        float y = y0 - p.y;
        float r = sqrtf(x * x + y * y) + 1e-20;
        if (r < 1.0f) {
            repulsion += repulsion_f(r, c_rep);
        }
        attraction += peak_f(r, mu_k, sigma_k, w_k);
    }

    // Calculate the new value for this cell
    d_world[j * width + i] = -repulsion + peak_f(attraction, mu_g, sigma_g);

    // Transform the value into something that is between 0 and 1
    d_world[j * width + i] = d_world[j * width + i] + 0.1f;
}
// CUDA kernel for solving the heat equation with shared memory
__global__ void
particleSolver(float3* d_particles)
{
    int n = blockIdx.x * blockDim.x + threadIdx.x;

    for (int step = 0; step < 1; ++step) {
        float2 r_grad = make_float2(0.0f, 0.0f);
        float2 u_grad = make_float2(0.0f, 0.0f);
        float u = peak_f(0.0f, mu_k, sigma_k, w_k);
        float repulsion = repulsion_f(0.0f, c_rep);

        for (int i = 0; i < NUM_PARTICLES; i++) {
            if (i == n) continue;
            float3 p1 = d_particles[n];
            float3 p2 = d_particles[i];
            float rx = p1.x - p2.x;
            float ry = p1.y - p2.y;
            float r = sqrtf(rx * rx + ry * ry) + 1e-20;
            rx /= r;
            ry /= r;

            if (r < 1.0) {
                float repulsion_grad = repulsion_fd(r, c_rep);
                r_grad.x += rx * repulsion_grad;
                r_grad.y += ry * repulsion_grad;
                repulsion += repulsion_f(r, c_rep);
            }

            float dk = peak_fd(r, mu_k, sigma_k, w_k);
            u_grad.x += rx * dk;
            u_grad.y += ry * dk;
            u += peak_f(r, mu_k, sigma_k, w_k);
        }

        float dg = peak_fd(u, mu_g, sigma_g);

        float2 v = make_float2(0.0f, 0.0f);
        v.x = dg * u_grad.x - r_grad.x;
        v.y = dg * u_grad.y - r_grad.y;

        float3 p = d_particles[n];
        p.x += v.x * dt;
        p.y += v.y * dt;

        d_particles[n].x = p.x;
        d_particles[n].y = p.y;
        d_particles[n].z = repulsion;
    }
}

static void
DrawCircle(float cx, float cy, float r, int num_segments)
{
    glBegin(GL_LINE_LOOP);
    for (int ii = 0; ii < num_segments; ii++) {
        float theta = 2.0f * 3.1415926f * float(ii) / float(num_segments);  // get the current angle
        float x = r * cosf(theta);  // calculate the x component
        float y = r * sinf(theta);  // calculate the y component
        glVertex2f(x + cx, y + cy);  // output vertex
    }
    glEnd();
}

// OpenGL display function
void
display()
{
    // Call the kernel to update the simulation
    particleSolver<<<NUM_PARTICLES, 1>>>(d_particlePositions);
    // Synchronize to ensure the kernel has finished
    cudaDeviceSynchronize();

    energySolver<<<dim3(NUM_BLOCKS_X, NUM_BLOCKS_Y), dim3(NUM_THREADS_X, NUM_THREADS_Y)>>>(
            d_world,
            d_particlePositions,
            WIDTH,
            HEIGHT);
    // Synchronize to ensure the kernel has finished
    cudaDeviceSynchronize();

    glClear(GL_COLOR_BUFFER_BIT);
    // Map the CUDA buffer to OpenGL
    float* d_mapped_buffer = nullptr;
    size_t num_bytes;
    cudaGraphicsMapResources(1, &cudaGLResource, 0);
    cudaGraphicsResourceGetMappedPointer((void**)&d_mapped_buffer, &num_bytes, cudaGLResource);

    // Copy data from d_output to the OpenGL PBO
    cudaMemcpy(d_mapped_buffer, d_world, WIDTH * HEIGHT * sizeof(float), cudaMemcpyDeviceToDevice);

    // Unmap the CUDA buffer from OpenGL
    cudaGraphicsUnmapResources(1, &cudaGLResource, 0);

    // If a non-zero named buffer object is bound to the
    // GL_PIXEL_UNPACK_BUFFER target (see main function) while a block of
    // pixels is specified, data is treated as a byte offset into the buffer
    // object's data store.
    glDrawPixels(WIDTH, HEIGHT, GL_RED, GL_FLOAT, 0);  // Note the use of nullptr

    // Copy the particle data
    cudaMemcpy(
            particlePositions,
            d_particlePositions,
            NUM_PARTICLES * sizeof(float3),
            cudaMemcpyDeviceToHost);

    // Render the particles using OpenGl as circles of radius 1 centered at the particle position
    for (int i = 0; i < NUM_PARTICLES; ++i) {
        float3 p = particlePositions[i];
        DrawCircle(p.x / SIZE * 2, p.y / SIZE * 2, c_rep / (p.z * 5 * SIZE), 10);
    }

    glutSwapBuffers();

    // Request the display function to be called
    glutPostRedisplay();
}

int
main(int argc, char** argv)
{
    // Initialize OpenGL and create a window
    glutInit(&argc, argv);
    glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE);
    glutInitWindowSize(WIDTH, HEIGHT);
    glutCreateWindow("Heat Equation Simulation");

    // Initialize GLEW
    GLenum err = glewInit();
    if (err != GLEW_OK) {
        std::cerr << "GLEW initialization error: " << glewGetErrorString(err) << std::endl;
        return EXIT_FAILURE;
    }

    // Set up OpenGL context and callbacks
    glutDisplayFunc(display);

    // Initialize CUDA device memory
    cudaMalloc((void**)&d_world, sizeof(float) * WIDTH * HEIGHT);
    cudaMalloc((void**)&d_particlePositions, sizeof(float3) * NUM_PARTICLES);

    // Initialize input data with initial conditions
    {
        std::vector<float3> pos;
        pos.reserve(NUM_PARTICLES);
        for (int i = 0; i < NUM_PARTICLES; ++i) {
            pos.emplace_back(make_float3(
                    (static_cast<float>(rand()) / RAND_MAX - 0.5) * SIZE,
                    (static_cast<float>(rand()) / RAND_MAX - 0.5) * SIZE,
                    1.0f));
        }

        cudaMemcpy(d_particlePositions, pos.data(), NUM_PARTICLES * sizeof(float3), cudaMemcpyHostToDevice);
    }

    // Initialize CUDA-OpenGL interoperability
    GLuint cudaGLBuffer;
    glGenBuffers(1, &cudaGLBuffer);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, cudaGLBuffer);
    glBufferData(GL_PIXEL_UNPACK_BUFFER, WIDTH * HEIGHT * sizeof(float), nullptr, GL_DYNAMIC_DRAW);
    cudaGraphicsGLRegisterBuffer(&cudaGLResource, cudaGLBuffer, cudaGraphicsMapFlagsNone);

    // Start the OpenGL main loop
    glutMainLoop();

    // Clean up CUDA and OpenGL resources
    cudaGraphicsUnregisterResource(cudaGLResource);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, cudaGLBuffer);
    glDeleteBuffers(1, &cudaGLBuffer);
    glutDestroyWindow(glutGetWindow());
    cudaFree(d_particlePositions);

    return EXIT_SUCCESS;
}
