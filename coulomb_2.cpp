//==============================================================================\
// PROGRAM TO CALCULATE THE TRAJECTORY, ELECTRIC AND MAGNETIC FIELDS OF N CHARGES\
//        Version 8.2 (to include many charges) -- by: Agenor (2025)             /
//==============================================================================/

#include <iostream>
#include <vector>
#include <cmath>
#include <fstream>
#include <mpi.h>
#include <chrono>
#include <algorithm>
#include <random>
#include <omp.h>

struct Particle {
    double x, y, z, vx, vy, vz, q, m;
    double Ex, Ey, Ez, Bx, By, Bz;
};

// Physical constants
const double k = 8.987551787e9;
const double mu0 = 4 * M_PI * 1e-7;
const double dt = 1e-6;
const double L = 1.0;
const double soft = 0.1;
const double velocity_limit = 1e3;  // Velocity limit constant


// RK4 helper structure
struct RK4State {
    double x, y, z;
    double vx, vy, vz;
};

MPI_Datatype create_mpi_particle_type() {
    MPI_Datatype MPI_Particle;
    int blocklengths[14] = {1,1,1,1,1,1,1,1,1,1,1,1,1,1};
    MPI_Datatype types[14] = {MPI_DOUBLE,MPI_DOUBLE,MPI_DOUBLE,
                             MPI_DOUBLE,MPI_DOUBLE,MPI_DOUBLE,
                             MPI_DOUBLE,MPI_DOUBLE,MPI_DOUBLE,
                             MPI_DOUBLE,MPI_DOUBLE,MPI_DOUBLE,
                             MPI_DOUBLE,MPI_DOUBLE};
    MPI_Aint offsets[14];
    
    offsets[0] = offsetof(Particle, x);
    offsets[1] = offsetof(Particle, y);
    offsets[2] = offsetof(Particle, z);
    offsets[3] = offsetof(Particle, vx);
    offsets[4] = offsetof(Particle, vy);
    offsets[5] = offsetof(Particle, vz);
    offsets[6] = offsetof(Particle, q);
    offsets[7] = offsetof(Particle, m);
    offsets[8] = offsetof(Particle, Ex);
    offsets[9] = offsetof(Particle, Ey);
    offsets[10] = offsetof(Particle, Ez);
    offsets[11] = offsetof(Particle, Bx);
    offsets[12] = offsetof(Particle, By);
    offsets[13] = offsetof(Particle, Bz);
    
    MPI_Type_create_struct(14, blocklengths, offsets, types, &MPI_Particle);
    MPI_Type_commit(&MPI_Particle);
    
    return MPI_Particle;
}

void calculate_acceleration(const Particle& p, double& ax, double& ay, double& az) {
    const double q_over_m = p.q / p.m;
    
    ax = q_over_m * (p.Ex + (p.vy * p.Bz - p.vz * p.By));
    ay = q_over_m * (p.Ey + (p.vz * p.Bx - p.vx * p.Bz));
    az = q_over_m * (p.Ez + (p.vx * p.By - p.vy * p.Bx));
}

void calculate_forces_direct(std::vector<Particle>& particles, int start, int end, int world_rank) {
    #pragma omp parallel for
    for(int i = start; i < end; i++) {
        particles[i].Ex = particles[i].Ey = particles[i].Ez = 0;
        particles[i].Bx = particles[i].By = particles[i].Bz = 0;
        
        for(int j = 0; j < particles.size(); j++) {
            if(i == j) continue;
            
            double dx = particles[j].x - particles[i].x;
            double dy = particles[j].y - particles[i].y;
            double dz = particles[j].z - particles[i].z;
            
            // Periodic boundary conditions
            dx -= L * round(dx/L);
            dy -= L * round(dy/L);
            dz -= L * round(dz/L);
            
            double r2 = dx*dx + dy*dy + dz*dz;
            // Apply softening
            if (r2 < soft*soft) {
                r2 = soft*soft;
            }
            
            double r = sqrt(r2);
            double k_over_r3 = k / (r2*r);
            double mu0_over_4pi_r3 = mu0 * particles[j].q / (4 * M_PI * r2*r);
            
            // Electric field
            particles[i].Ex += particles[j].q * dx * k_over_r3;
            particles[i].Ey += particles[j].q * dy * k_over_r3;
            particles[i].Ez += particles[j].q * dz * k_over_r3;
            
            // Magnetic field with velocity limiting
            double vx = std::min(std::max(particles[j].vx, -velocity_limit), velocity_limit);
            double vy = std::min(std::max(particles[j].vy, -velocity_limit), velocity_limit);
            double vz = std::min(std::max(particles[j].vz, -velocity_limit), velocity_limit);
            
            double v_cross_r_x = dy * vz - dz * vy;
            double v_cross_r_y = dz * vx - dx * vz;
            double v_cross_r_z = dx * vy - dy * vx;
            
            particles[i].Bx += mu0_over_4pi_r3 * v_cross_r_x;
            particles[i].By += mu0_over_4pi_r3 * v_cross_r_y;
            particles[i].Bz += mu0_over_4pi_r3 * v_cross_r_z;
        }
    }
}

void update_particles_rk4(std::vector<Particle>& particles, int start, int end, double dt) {
    #pragma omp parallel for
    for(int i = start; i < end; i++) {
        RK4State k1, k2, k3, k4;
        double ax, ay, az;
        
        // Stage 1
        calculate_acceleration(particles[i], k1.vx, k1.vy, k1.vz);
        k1.x = particles[i].vx;
        k1.y = particles[i].vy;
        k1.z = particles[i].vz;
        
        // Stage 2
        Particle temp = particles[i];
        temp.vx += 0.5 * dt * k1.vx;
        temp.vy += 0.5 * dt * k1.vy;
        temp.vz += 0.5 * dt * k1.vz;
        temp.x += 0.5 * dt * k1.x;
        temp.y += 0.5 * dt * k1.y;
        temp.z += 0.5 * dt * k1.z;
        
        calculate_acceleration(temp, k2.vx, k2.vy, k2.vz);
        k2.x = temp.vx;
        k2.y = temp.vy;
        k2.z = temp.vz;
        
        // Stage 3
        temp = particles[i];
        temp.vx += 0.5 * dt * k2.vx;
        temp.vy += 0.5 * dt * k2.vy;
        temp.vz += 0.5 * dt * k2.vz;
        temp.x += 0.5 * dt * k2.x;
        temp.y += 0.5 * dt * k2.y;
        temp.z += 0.5 * dt * k2.z;
        
        calculate_acceleration(temp, k3.vx, k3.vy, k3.vz);
        k3.x = temp.vx;
        k3.y = temp.vy;
        k3.z = temp.vz;
        
        // Stage 4
        temp = particles[i];
        temp.vx += dt * k3.vx;
        temp.vy += dt * k3.vy;
        temp.vz += dt * k3.vz;
        temp.x += dt * k3.x;
        temp.y += dt * k3.y;
        temp.z += dt * k3.z;
        
        calculate_acceleration(temp, k4.vx, k4.vy, k4.vz);
        k4.x = temp.vx;
        k4.y = temp.vy;
        k4.z = temp.vz;
        
        // Final update
        particles[i].vx += (dt/6.0) * (k1.vx + 2*k2.vx + 2*k3.vx + k4.vx);
        particles[i].vy += (dt/6.0) * (k1.vy + 2*k2.vy + 2*k3.vy + k4.vy);
        particles[i].vz += (dt/6.0) * (k1.vz + 2*k2.vz + 2*k3.vz + k4.vz);
        
        particles[i].x += (dt/6.0) * (k1.x + 2*k2.x + 2*k3.x + k4.x);
        particles[i].y += (dt/6.0) * (k1.y + 2*k2.y + 2*k3.y + k4.y);
        particles[i].z += (dt/6.0) * (k1.z + 2*k2.z + 2*k3.z + k4.z);
        
        // Periodic boundary conditions
        particles[i].x = fmod(fmod(particles[i].x, L) + L, L);
        particles[i].y = fmod(fmod(particles[i].y, L) + L, L);
        particles[i].z = fmod(fmod(particles[i].z, L) + L, L);
    }
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    
    int world_size, world_rank;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    
    const int N = 10000;
    const int steps = 10000;
    const int output_interval = 10;
    
    // Work distribution
    int particles_per_proc = N / world_size;
    int remainder = N % world_size;
    int start = world_rank * particles_per_proc + (world_rank < remainder ? world_rank : remainder);
    int end = start + particles_per_proc + (world_rank < remainder ? 1 : 0);
    
    MPI_Datatype MPI_Particle = create_mpi_particle_type();
    
    // Particle initialization
    std::vector<Particle> particles(N);
    if(world_rank == 0) {
        std::mt19937 gen(42);
        std::uniform_real_distribution<double> pos_dist(0.0, L);
        std::uniform_real_distribution<double> charge_dist(-1.0, 1.0);
        std::uniform_real_distribution<double> vel_dist(-0.1, 0.1);
        
        for(int i = 0; i < N; i++) {
            particles[i].x = pos_dist(gen);
            particles[i].y = pos_dist(gen);
            particles[i].z = pos_dist(gen);
            particles[i].vx = vel_dist(gen);
            particles[i].vy = vel_dist(gen);
            particles[i].vz = vel_dist(gen);
            particles[i].q = charge_dist(gen);
            particles[i].m = 1.0;
        }
    }
    
    // Broadcast particles to all processes
    MPI_Bcast(particles.data(), N, MPI_Particle, 0, MPI_COMM_WORLD);
    
    // Output file
    std::ofstream outfile;
    if(world_rank == 0) {
        outfile.open("trajectories_fields.dat");
        std::cout << "Starting simulation with " << N << " particles for " << steps << " steps\n";
        std::cout << "Using " << world_size << " MPI processes\n";
    }
    
    auto start_time = std::chrono::high_resolution_clock::now();
    
    for(int step = 0; step < steps; step++) {
        // Calculate forces
        calculate_forces_direct(particles, start, end, world_rank);
        
        // Update particles
        update_particles_rk4(particles, start, end, dt);
        
        // Synchronize data
        MPI_Allgather(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL,
                     particles.data(), particles_per_proc, MPI_Particle,
                     MPI_COMM_WORLD);
        
        // Debug print after communication
        if(step % output_interval == 0 && world_rank == 0) {
            for(int i = 0; i < 5; i++) {
                std::cout << "POST-COMM Step " << step << " Part " << i 
                          << " B fields: " << particles[i].Bx << " " 
                          << particles[i].By << " " << particles[i].Bz << std::endl;
            }
        }
        
        // Output data
        if(world_rank == 0 && step % output_interval == 0) {
            for(const auto& p : particles) {
                outfile << p.x << " " << p.y << " " << p.z << " "
                        << p.q << " "
                        << p.Ex << " " << p.Ey << " " << p.Ez << " "
                        << p.Bx << " " << p.By << " " << p.Bz << "\n";
            }
            outfile << "\n";
            outfile.flush();
        }
    }
    
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
    
    if(world_rank == 0) {
        outfile.close();
        std::cout << "Simulation completed in " << duration.count()/1000.0 << " seconds\n";
        std::cout << "Data saved to trajectories_fields.dat\n";
        
        // Final verification
        std::cout << "\nFinal verification for first particle:\n";
        std::cout << "Position: " << particles[0].x << " " << particles[0].y << " " << particles[0].z << "\n";
        std::cout << "Velocity: " << particles[0].vx << " " << particles[0].vy << " " << particles[0].vz << "\n";
        std::cout << "E Field: " << particles[0].Ex << " " << particles[0].Ey << " " << particles[0].Ez << "\n";
        std::cout << "B Field: " << particles[0].Bx << " " << particles[0].By << " " << particles[0].Bz << "\n";
    }
    
    MPI_Type_free(&MPI_Particle);
    MPI_Finalize();
    return 0;
}
