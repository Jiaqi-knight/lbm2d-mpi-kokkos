#include <mpi.h>
#include <ostream>
#include <stdlib.h>
#include <memory>

#include "functors.h"
#include "params.h"
#include "output.h"

void update(DistributionField fB, DistributionField fA, ScalarField u, ScalarField v, ScalarField rho, Params &params, const int step);

int main(int narg, char *arg[]) {

  int actual_threads;

  MPI_Init_thread(NULL, NULL, MPI_THREAD_FUNNELED, &actual_threads);

  Kokkos::initialize(narg, arg);

  Params params;

  MPI_Comm_size(MPI_COMM_WORLD, &params.num_proc);

  MPI_Comm_rank(MPI_COMM_WORLD, &params.rank);

  MPI_Status statuses[4];

  MPI_Request req[4];

  std::unique_ptr < Output > o(new Output());

  {
    // params
    params.nx = atoi(arg[1]);
    params.ny = atoi(arg[2]);

    params.re = atof(arg[3]);
    params.tol = atof(arg[4]);
    params.max_steps = atoi(arg[5]);
    params.output_rate = atoi(arg[6]);

    params.u_lid = 0.1;
    params.nu = params.u_lid * double(params.ny - 2) / params.re;
    params.tau = 3. * params.nu + 0.5;

    int tmp = ((params.ny - 2) / params.num_proc + 2);
    int nx = params.nx;
    int ny = tmp;

    if (params.rank == 0) {

      ny = params.ny - (tmp - 2) * (params.num_proc - 1);
    }

    int buff_size = nx * 3;

    // allocate memory on device
    DistributionField fA("fA", ny, nx);
    DistributionField fB("fB", ny, nx);
    DistributionField fC = fA;

    BoundaryDistributionField fT_send("fT_send", nx);
    BoundaryDistributionField fT_recv("fT_recv", nx);
    BoundaryDistributionField fB_send("fB_send", nx);
    BoundaryDistributionField fB_recv("fB_recv", nx);

    ScalarField rho("rho", ny, nx);
    ScalarField u("u", ny, nx);
    ScalarField v("v", ny, nx);

    //create macro mirrors
    ScalarField::HostMirror h_u = Kokkos::create_mirror_view(u);
    ScalarField::HostMirror h_v = Kokkos::create_mirror_view(v);
    ScalarField::HostMirror h_rho = Kokkos::create_mirror_view(rho);

    // create xfer mirrors
    BoundaryDistributionField::HostMirror h_fT_send = Kokkos::create_mirror_view(fT_send);
    BoundaryDistributionField::HostMirror h_fT_recv = Kokkos::create_mirror_view(fT_recv);
    BoundaryDistributionField::HostMirror h_fB_send = Kokkos::create_mirror_view(fB_send);
    BoundaryDistributionField::HostMirror h_fB_recv = Kokkos::create_mirror_view(fB_recv);

    // initialize values
    Kokkos::parallel_for("load_state", range_2d( { 0, 0 }, { fA.extent(0), fA.extent(1) }), load_state(fA, fB, u, v, rho));

    // write to output file
    o->write_view("output/u", h_u, params);
    o->write_view("output/v", h_v, params);
    o->write_view("output/rho", h_rho, params);
    o->frame += 1;

    printf("Solving lid driven cavity (rank = %i of %i, Re = %.2e, tau = %.2e, domain [%i x %i] , RAM (MB) = %.1f)...\n", params.rank + 1, params.num_proc, params.re, params.tau,
        nx - 2, ny - 2, (2. * 9. + 3.) * 8. * double(ny - 2) * double (nx-2) / (1024. * 1024.));

    Kokkos::Timer timer;

    double time_comm = 0.0;

    int step = 0;
    int converged = 0;

    while (step < params.max_steps && !converged) {

      // collide-stream-bounceback
      update(fB, fA, u, v, rho, params, step);

      Kokkos::Timer timer2;

      // MPI calls
      if (params.num_proc > 1) {

        int rank_above = params.rank + 1;
        int rank_below = params.rank - 1;

        if (rank_above > (params.num_proc - 1)) {
          rank_above = 0;
        }

        if (rank_below < 0) {
          rank_below = params.num_proc - 1;
        }

        // populate xfer buffers

        // if MPI w/ CPUs, don't launch any OpenMP threads
//        for (int j = 1; j < (nx - 1); ++j) {
//
//          fT_send(j, 0) = fB(ny - 1, j, 5);
//          fT_send(j, 1) = fB(ny - 1, j, 2);
//          fT_send(j, 2) = fB(ny - 1, j, 6);
//
//          fB_send(j, 0) = fB(0, j, 7);
//          fB_send(j, 1) = fB(0, j, 4);
//          fB_send(j, 2) = fB(0, j, 8);
//        }

        // if MPI w/ GPUS
        Kokkos::parallel_for("populate_send_buffers",range_1d(1, nx - 1), populate_send_buffers(fB, fT_send, fB_send, ny);

        // copy "send" buffer from device to host
        // Kokkos::deep_copy(h_fT_send, fT_send); // should not need with Cuda-aware MPI
        // Kokkos::deep_copy(h_fB_send, fB_send);

        // receive from slab below
        //MPI_Irecv(h_fB_recv.data(), buff_size, MPI_DOUBLE, rank_below, 0, MPI_COMM_WORLD, &req[1]); // 5,2,6
        MPI_Irecv(fB_recv.data(), buff_size, MPI_DOUBLE, rank_below, 0, MPI_COMM_WORLD, &req[1]); // 5,2,6

        // receive from slab above
        //MPI_Irecv(h_fT_recv.data(), buff_size, MPI_DOUBLE, rank_above, 0, MPI_COMM_WORLD, &req[3]); // 7, 4, 8

        // send to slab above
        //MPI_Isend(h_fT_send.data(), buff_size, MPI_DOUBLE, rank_above, 0, MPI_COMM_WORLD, &req[0]);	// 5,2,6

        // send to slab below
        //MPI_Isend(h_fB_send.data(), buff_size, MPI_DOUBLE, rank_below, 0, MPI_COMM_WORLD, &req[2]); // 7, 4, 8

        MPI_Waitall(4, req, statuses);

        // copy "recv" buffer from host to device
        Kokkos::deep_copy(fT_recv, h_fT_recv);
        Kokkos::deep_copy(fB_recv, h_fB_recv);

        // use recv buffer data to set distributions for slabs 1 ...N-2
        for (int j = 1; j < (nx - 1); ++j) {

          if (params.rank > 0 and params.rank < (params.num_proc - 1)) {

            if (j == 1) {

              fB(ny - 2, j, 7) = fT_recv(j, 0);
              fB(ny - 2, j, 4) = fT_recv(j, 1);

              fB(1, j, 2) = fB_recv(j, 1);
              fB(1, j, 6) = fB_recv(j, 2);

            } else if (j == nx - 2) {

              fB(ny - 2, j, 4) = fT_recv(j, 1);
              fB(ny - 2, j, 8) = fT_recv(j, 2);

              fB(1, j, 5) = fB_recv(j, 0);
              fB(1, j, 2) = fB_recv(j, 1);

            } else {

              fB(ny - 2, j, 7) = fT_recv(j, 0);
              fB(ny - 2, j, 4) = fT_recv(j, 1);
              fB(ny - 2, j, 8) = fT_recv(j, 2);

              fB(1, j, 5) = fB_recv(j, 0);
              fB(1, j, 2) = fB_recv(j, 1);
              fB(1, j, 6) = fB_recv(j, 2);
            }

          } else if (params.rank == 0) {

            if (j == 1) {

              fB(ny - 2, j, 7) = fT_recv(j, 0);
              fB(ny - 2, j, 4) = fT_recv(j, 1);

            } else if (j == (nx - 2)) {

              fB(ny - 2, j, 4) = fT_recv(j, 1);
              fB(ny - 2, j, 8) = fT_recv(j, 2);

            } else {

              fB(ny - 2, j, 7) = fT_recv(j, 0);
              fB(ny - 2, j, 4) = fT_recv(j, 1);
              fB(ny - 2, j, 8) = fT_recv(j, 2);

            }

          } else if (params.rank == (params.num_proc - 1)) {

            if (j == 1) {

              fB(1, j, 2) = fB_recv(j, 1);
              fB(1, j, 6) = fB_recv(j, 2);

            } else if (j == (nx - 2)) {

              fB(1, j, 5) = fB_recv(j, 0);
              fB(1, j, 2) = fB_recv(j, 1);

            } else {

              fB(1, j, 5) = fB_recv(j, 0);
              fB(1, j, 2) = fB_recv(j, 1);
              fB(1, j, 6) = fB_recv(j, 2);
            }
          }
        }
      }

      time_comm += timer2.seconds();

      // distributions are updated, compute macroscopic and do steady state check
      if ((step + 1) % params.output_rate == 0) {

        Kokkos::parallel_for("compute_macroscopic", range_2d( { 1, 1 }, { fA.extent(0) - 1, fA.extent(1) - 1 }), compute_macroscopic(fB, u, v, rho));

        converged = 1;

        Kokkos::parallel_reduce("check_if_steady_state", range_2d( { 1, 1 }, { fA.extent(0) - 1, fA.extent(1) - 1 }),
            is_steady_state(fA, u, v, rho, params.tol), Kokkos::BAnd<int>(converged));

        // MPI calls
        if (params.num_proc > 1) {
          int converged_globally = 0;

          MPI_Allreduce(&converged, &converged_globally, 1, MPI_INT, MPI_BAND, MPI_COMM_WORLD);

          converged = converged_globally;
        }
      }

      // swap pointers
      fC = fA;
      fA = fB;
      fB = fC;

      // output macroscopic variables
      if ((step + 1) % params.output_rate == 0) {

        printf("...output step = %i, rank = %i\n", step + 1, params.rank);

        // deep copy from device to host
        Kokkos::deep_copy(h_u, u);
        Kokkos::deep_copy(h_v, v);
        Kokkos::deep_copy(h_rho, rho);

        // write to output file
        o->write_view("output/u", h_u, params);
        o->write_view("output/v", h_v, params);
        o->write_view("output/rho", h_rho, params);

        o->frame += 1;
      }

      step += 1;
    }

    double time = timer.seconds();

    double site_updates = double(nx - 2) * double(ny - 2) * double(step) / (1000. * 1000.);
    double msus = site_updates / time;
    double bandwidth = msus * 1000. * 1000. * 2. * 9. * 8. / (1024. * 1024. * 1024.);

    printf("rank = %i, MLUPS = %.1f, GB/s = %.1f, comm_time/total_time = %.3e\n", params.rank, msus, bandwidth, time_comm / time);

    if (converged && params.rank == 0) {
      printf("Solution converged to steady state tolerance of %.3e\n", params.tol);
    } else if (!converged && params.rank == 0) {
      printf("Solution did not converged within %i steps\n", params.max_steps);
    }

    // MPI calls
    if (params.num_proc > 1) {

      double tot_msus = 0.0;

      MPI_Reduce(&msus, &tot_msus, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

      if (params.rank == 0) {
        printf("MLUPs = %.1f\n", tot_msus);
      }
    }
  }

  Kokkos::finalize();

  MPI_Finalize();

  exit(EXIT_SUCCESS);
}

void update(DistributionField fB, DistributionField fA, ScalarField u, ScalarField v, ScalarField rho, Params &params, const int step) {

  const int ny = fA.extent(0);
  const int nx = fA.extent(1);

  Kokkos::parallel_for("collision_streaming", range_2d( { 1, 1 }, { ny - 1, nx - 1 }), collide_stream(fB, fA, params));
  Kokkos::parallel_for("bb_west", range_1d(1, ny - 1), bb_west(fB));
  Kokkos::parallel_for("bb_east", range_1d(1, ny - 1), bb_east(fB));

  if (params.num_proc > 1) {

    if (params.rank == 0) {

      Kokkos::parallel_for("bb_south", range_1d(1, nx - 1), bb_south(fB));

    } else if (params.rank == params.num_proc - 1) {

      Kokkos::parallel_for("bb_north", range_1d(1, nx - 1), bb_north(fB, params.u_lid));

    }
  } else {

    Kokkos::parallel_for("bb_south", range_1d(1, nx - 1), bb_south(fB));

    Kokkos::parallel_for("bb_north", range_1d(1, nx - 1), bb_north(fB, params.u_lid));
  }
}

