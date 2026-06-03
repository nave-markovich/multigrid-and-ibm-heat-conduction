# multigrid-and-ibm-heat-conduction
2D heat conduction solver in a square plate with a central hole. Implements Gauss-Seidel, Multigrid (V-cycle), BiCgStab preconditioning, body-fitted triangular mesh (LU direct method), and Immersed Boundary Method (IBM) with Schur complement for complex geometries.

# Advanced Numerical Methods for CFD: 2D Heat Conduction Solver

This repository contains the final project for the **Advanced Numerical Methods** course at **Ben-Gurion University of the Negev**, developed by **Avi Kutai** and **Nave Markovich**. 

The project implements and compares various advanced finite volume and immersed boundary techniques to solve steady and unsteady 2D conductive heat transfer equations across severe material discontinuities ($k_1 = 10^{-3}$ and $k_2 = 100$)[cite: 1, 2].

For full mathematical derivations, convergence analysis, and physical discussions, please reference **Project.pdf**.

---

## Physical Model & Governing Equation
The conductive heat transfer within a 2D square plate ($-0.5 \le x, y \le 0.5$)[cite: 1] enclosing a central circular obstacle ($R = 0.2$)[cite: 1] is governed by the energy equation:

$$\frac{\partial}{\partial x}\left(k\frac{\partial T}{\partial x}\right) + \frac{\partial}{\partial y}\left(k\frac{\partial T}{\partial y}\right) = \frac{\partial T}{\partial t}$$

### Managing the $10^5$ Conductivity Discontinuity
To strictly satisfy heat flux continuity across the sharp interface of the two materials, a simple arithmetic mean fails physically[cite: 2]. Instead, we employ a **Harmonic Mean** to evaluate interface conductivities[cite: 2]:
$$k_{\text{face}} = \frac{2k_P k_E}{k_P + k_E}$$
This preserves the physical integrity of the insulating boundary layer without artificial numerical smearing[cite: 2].

---

## Project Structure & Methods Implemented

### Part 1: Homogeneous Geometry (Steady State)
Solved on a $200 \times 200$ cell-centered Finite Volume (FVM) grid[cite: 1, 2]:
* **Line Gauss-Seidel Smoother:** Sweeps in the x-direction where each vertical line is resolved as a tridiagonal system via the Thomas algorithm[cite: 2].
* **Geometric Multigrid (V-Cycle):** Optimized using a 4-point conservative average for restriction and piecewise-constant expansion for prolongation to prevent artificial conductivity blending on coarser grids[cite: 2].
* **Krylov Subspace Acceleration:** Utilizing a single Multigrid V-cycle as a left preconditioner for MATLAB's built-in **BiCgStab** solver[cite: 1, 2].
* **Complexity Analysis:** Log-log scaling analysis ($t(N) \sim N^\beta$) proving near-optimal theoretical complexity ($\beta \approx 1.94 - 1.97$)[cite: 2].

### Part 2: Complex Geometry with Central Hole ($R = 0.2$)
* **Unsteady Boundary-Fitted Mesh (LU Direct Method):** Solved on an $80 \times 80$ non-orthogonal triangular mesh[cite: 1, 2]. Non-orthogonality is handled via the **Minimum Correction Method** coupled with Green-Gauss gradient reconstruction in deferred correction form[cite: 2]. Time integration utilizes a fully implicit Backward Euler scheme[cite: 2].
* **Implicit Immersed Boundary Method (IBM):** Implemented on a $200 \times 200$ structured Cartesian grid[cite: 1, 2]. Dirichlet boundary conditions on the hole ($T=2$)[cite: 1, 2] are handled via Lagrangian markers with a 4-point regularized Peskin delta kernel, yielding a saddle-point system solved via **Schur Complement formulation**[cite: 2].
* **Explicit Direct Forcing IBM (Insulated Hole):** Implements a zero normal heat flux (Neumann BC) using a **Probe & Ghost point strategy**[cite: 2]. Interpolation and spreading operations are optimized using pre-computed sparse delta-function matrices to eliminate heavy pointwise loops inside the time-stepping cycle[cite: 2].


### Simulation Results
Here is an example of the computed temperature distribution and grid configuration from the simulation:
* <img width="548" height="352" alt="image" src="https://github.com/user-attachments/assets/4981fe7d-836f-48fa-938e-b719ad77ba18" />


---

## Computational Performance & Optimization Highlights
* **Vectorized Architecture:** Fully vectorized residual computations and structural matrix-free operations within the recursive Multigrid steps using MATLAB's `repelem`[cite: 2].
* **Pre-Factorization:** Both the Eulerian operators and Schur matrices are pre-factorized via LU decomposition outside the time loop, drastically reducing per-step computational overhead[cite: 2].

## Software Requirements
* **MATLAB** (Solvers, Custom Meshing, and Post-Processing)[cite: 1, 2]
