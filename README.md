# The Characteristic Mapping Method

Python implementation of the Characteristic Mapping Method (CMM) on some two-dimensional domains. The CMM is a semi-Lagrangian scheme which computes the evolution of the inverse map through a spatiotemporal discretization formed by a composition of sub-interval flows. These are each represented as spline interpolants computing using the gradient-augmented level set method. This repository catalogues a collection of functions and tests relevant to this computation for linear advection and the incompressible Euler equations on the two-torus and on the sphere. It contains all relevant code needed to reproduce the figures and data in the papers [1],[2]. The repository will be actively maintained and extended to solve other types of flows in different domains.


# Dependencies
Standard dependencies come from the `numpy` and `scipy` packages. Specific dependencies used in the implemenation include:

- The spherical harmonic transforms are performed using the `pyssht` [library](https://pypi.org/project/pyssht/)
- The spherical mesh generation is performed using the `stripy` [package](https://pypi.org/project/stripy/)
- The point in triangle querying is facilitate a single function (point_mesh_squared_distance) from the [python binding](https://libigl.github.io/libigl-python-bindings/) to the `lib-igl` package

All dependencies can be installed via `pip`. <u> Note <u> the dependencies will likely change in the future. 

# Code structure

All scripts relevant to the numerical time-stepping algorithm are contained in the cmm/core folder. All convergence tests for linear advection and incompressible Euler are contained in the cmm/tests/ folder. 
