# Lenia particle simulation

This is a project for simulating Lenia particles using CUDA and OpenGL. The
simulation involves particles interacting with each other via the Lenia potential
The simulation is visualized using OpenGL.

## Table of Contents
- [Introduction](#introduction)
- [Prerequisites](#prerequisites)
- [Getting Started](#getting-started)
- [Usage](#usage)
- [License](#license)

## Introduction

The Lenia simulation project is a GPU-accelerated simulation that
models the behavior of particles in a 2D environment. The simulation calculates
the particle interactions over time, producing a visual representation of the system.

## Prerequisites

Before running the simulation, ensure you have the following prerequisites installed:

- [OpenGL](https://www.opengl.org/) - for rendering the simulation graphics.
- [GLEW](http://glew.sourceforge.net/) - the OpenGL Extension Wrangler Library.
- [freeglut](http://freeglut.sourceforge.net/) - an open-source alternative to the GLUT library for creating and managing windows.
- [CUDA](https://developer.nvidia.com/cuda-toolkit) - the NVIDIA GPU computing platform.

## Getting Started

To get started with the Heat Equation Simulation, follow these steps:

1. Clone this repository:

   ```bash
   git clone git@github.com:pablogsal/cuda_lenia.git
   ```

2. Navigate to the project directory:

   ```bash
   cd cuda_lenia
   ```

3. Build the project using a CMake. Make sure to link the necessary libraries (OpenGL, GLEW,
   CUDA).

   ```bash
   mkdir build
   cd build
   cmake ..
   make
   ```

4. Run the compiled executable:

   ```bash
   ./lenia_particles
   ```

## Usage

- Upon running the simulation, a window will appear showing the evolving of the particles.
- The simulation will continue until you close the window or terminate the program.
- You can adjust simulation parameters and initial conditions in the `main`
  function, such as the number of particles, simulation size, and simulation
  constants.

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
