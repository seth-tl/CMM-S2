<img align="left" height="75" width="75" src="./docs/assets/cmm-s2_logo.png">

# The Spherical Characteristic Mapping Method

`CMM-S2` is a Python implementation of the Characteristic Mapping Method (CMM) on a spherical geometry. The CMM is a semi-Lagrangian scheme which computes the evolution of the inverse map through a spatiotemporal discretization formed by a composition of sub-interval flows. These are each represented as spline interpolants computing using the gradient-augmented level set method. This repository catalogues a collection of functions and tests relevant to this computation for linear advection and the incompressible Euler equations on the two-torus and on the sphere. It contains all relevant code needed to reproduce the figures and data in the papers [1](https://www.sciencedirect.com/science/article/pii/S0021999122009688?casa_token=XLpApKjiy_wAAAAA:d0pBJ0JlQfz7WpwiINySp_ceZF8ECV9v8xHKZ9PWz3QP7bKiyutZBS1HfOcpuk8L5_JQXCtD3g) and [2](https://arxiv.org/pdf/2302.01205.pdf). The repository will be actively maintained and extended to solve other types of flows in different domains.

# Capabilities

The implementation is capable of simulating turbulent fluid dynamics on a rotating sphere in the form of the barotropic vorticity equations. It possess a number of unique resolution properties including the ability to upsample the solution with the correct statistics at subgrid scales and to coherently transport a multi-scale field. 

![image](./docs/assets/multi_jet_evolution_redo.png)
![image](./docs/assets/multi_jet_redo_zoom.png)


It further provides a base solver class for the transport equation, which can be ported into other solvers for tracer and flow map analysis. An interface for more general velocity field data and ERA5 data is in the works. 




# Dependencies
Standard dependencies come from the `numpy` and `scipy` packages. Specific dependencies used in the implemenation include:

- The spherical harmonic transforms are performed using the `pyssht` [library](https://pypi.org/project/pyssht/)
- The spherical mesh generation is performed using the `stripy` [package](https://pypi.org/project/stripy/)
- The point in triangle querying is facilitate a single function (point_mesh_squared_distance) from the [python binding](https://libigl.github.io/libigl-python-bindings/) to the `lib-igl` package

All dependencies can be installed via `pip`. <u> Note <u> the dependencies will likely change in the future. 

# Code structure

All scripts relevant to the numerical time-stepping algorithm are contained in the cmm/core folder. All convergence tests for linear advection and incompressible Euler are contained in the cmm/tests/ folder. 
