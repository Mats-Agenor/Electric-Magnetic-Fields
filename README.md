# charges
----------
compile with mpicxx -std=c++11 -O3 coulomb_2.cpp -o coulomb -lgsl -lgslcblas

execute with mpirun -np 20 ./coulomb

On a cluster with 20 MPI processes, this program can be run in 40-50 minutes.

----------
This C++ program is a parallelized particle simulation designed to model the motion of charged particles under self-consistent electromagnetic fields. It combines finite-time-step integration of Newton's laws with MPI for distributed memory parallelism and OpenMP for shared memory acceleration. Particles interact by directly computing electric and magnetic fields, without using grid-based methods. The goal is to compute the collective dynamics of many particles (e.g. 10,000) evolving over time, influenced by their mutual electromagnetic interactions.

Overview of Key Features
Data Structures and Constants:

Each particle is represented by a struct Particle containing the position, velocity, charge, mass, and electromagnetic field vectors (E and B).

Constants such as the Coulomb constant k, the magnetic permeability mu0, the time step dt, the domain size L, the softening length, and a speed limit are defined to control the simulation and physics stability.

Parallelism with MPI and OpenMP:

The code uses MPI to divide the workload among multiple processes. Each process computes interactions and updates for a distinct subset of the total particles.

MPI data types are carefully constructed to pass and collect particle structures between processes.

OpenMP parallelizes loops within each process to exploit multithreading on shared memory systems.

Initialization:

The master process (rank 0) initializes particles with random positions in a unit cube, random small velocities, random charges, and a fixed mass. These values ​​are passed to all processes.

Force Calculation:

For each particle, the electric field is calculated via pairwise Coulomb interactions, with smoothing applied at small separations to avoid singularities.

The magnetic field is calculated using a simplified Biot-Savart-like approach, where the contribution of each particle depends on its velocity and relative position.

A velocity limiter is used to limit particle velocities during the magnetic field calculation, improving numerical stability.

Periodic boundary conditions are imposed to emulate an infinite system, involving particle positions at the edges of the domain.

Integration Scheme (RK4):

The particle motion is integrated using the classical fourth-order Runge–Kutta (RK4) method.

The RK4 integrator is applied in a custom update_particles_rk4() function, using instantaneous accelerations calculated from the Lorentz force.

Positions are encapsulated within the domain after each update to maintain periodicity.

Simulation Loop:

The main loop runs for a user-defined number of time steps, updating fields and integrating particle states at each iteration.

After each time step, the updated particle states are synchronized using MPI_Allgather to ensure that all processes have access to all particle data before the next iteration.

At each output_interval step, the master process writes the complete particle state (positions, charge, E-field, B-field) to a file called trajectories_fields.dat.

Logging and Output:

The code includes regular console output to track the progress of the simulation, including sample magnetic field values ​​for debugging.

At the end of the run, the master process reports the total execution time and prints the final state of the first particle for basic verification.

Applications and Use Cases
This code can be used as a testbed for studying basic plasma physics or charged particle dynamics in electromagnetic fields, especially in the context of collisionless systems. It is suitable for educational purposes or cases where full particle-particle interactions are required and field interpolation or particle-in-cell methods are not yet needed. It is also designed to be extensible, allowing for future improvements such as adaptive timestepping, relativistic corrections, or GPU acceleration.
