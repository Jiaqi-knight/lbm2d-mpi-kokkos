#include <mpi.h>
#include <ostream>
#include <stdlib.h>

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

  Output o;

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

    int nx = params.nx;
    int tmp = (params.ny - 2) / params.num_proc;
    int rem = (params.ny - 2) % params.num_proc;

    // distribute remainder as evenly as possible
    if (params.rank < rem) {
      tmp += 1;
    }

    // add ghost layers;
    int ny = tmp + 2;

    int buff_size = nx * 3;

    // allocate memory on device
    DistributionField fA("fA", ny, nx);
    DistributionField fB("fB", ny, nx);
    DistributionField fC = fA;

    HaloDistField fT_send("fT_send", nx);
    HaloDistField fT_recv("fT_recv", nx);
    HaloDistField fB_send("fB_send", nx);
    HaloDistField fB_recv("fB_recv", nx);

    ScalarField rho("rho", ny, nx);
    ScalarField u("u", ny, nx);
    ScalarField v("v", ny, nx);

    //create macro mirrors
    ScalarField::HostMirror h_u = Kokkos::create_mirror_view(u);
    ScalarField::HostMirror h_v = Kokkos::create_mirror_view(v);
    ScalarField::HostMirror h_rho = Kokkos::create_mirror_view(rho);

    // initialize values
    Kokkos::parallel_for("load_state", range_2d( { 0, 0 }, { fA.extent(0), fA.extent(1) }), load_state(fA, fB, u, v, rho));

    // write to output file
    o.write_view("output/u", h_u, params);
    o.write_view("output/v", h_v, params);
    o.write_view("output/rho", h_rho, params);
    o.frame += 1;

    printf("Solving lid driven cavity (rank = %i of %i, Re = %.2e, tau = %.2e, domain [%i x %i] , RAM (MB) = %.1f)...\n", params.rank + 1, params.num_proc, params.re, params.tau,
        ny - 2, nx - 2, (2. * 9. + 3.) * 8. * double(ny - 2) * double (nx-2) / (1024. * 1024.));

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

        // TODO if MPI w/ CPUs, don't launch any OpenMP threads
        Kokkos::parallel_for("populate_send_buffers",range_1d(1, nx - 1), populate_send_buffers(fB, fT_send, fB_send, ny));

        // receive from slab below
        MPI_Irecv(fB_recv.data(), buff_size, MPI_DOUBLE, rank_below, 0, MPI_COMM_WORLD, &req[1]); // 5,2,6

        // receive from slab above
        MPI_Irecv(fT_recv.data(), buff_size, MPI_DOUBLE, rank_above, 0, MPI_COMM_WORLD, &req[3]); // 7, 4, 8

        // send to slab above
        MPI_Isend(fT_send.data(), buff_size, MPI_DOUBLE, rank_above, 0, MPI_COMM_WORLD, &req[0]);   // 5,2,6

        // send to slab below
        MPI_Isend(fB_send.data(), buff_size, MPI_DOUBLE, rank_below, 0, MPI_COMM_WORLD, &req[2]); // 7, 4, 8

        MPI_Waitall(4, req, statuses);

        // TODO if MPI w/ CPUs dont launch any OpenMP threads
        if (params.rank > 0 and params.rank < (params.num_proc - 1)) {
          Kokkos::parallel_for("load_from_recv_buffers_rank_k",range_1d(1, nx - 1), load_from_recv_buffers_rank_k(fB, fT_recv, fB_recv, ny, nx));
        } else if (params.rank == 0) {
          Kokkos::parallel_for("load_from_recv_buffers_rank_zero",range_1d(1, nx - 1), load_from_recv_buffers_rank_zero(fB, fT_recv, fB_recv, ny, nx));
        } else if (params.rank == (params.num_proc - 1)) {
          Kokkos::parallel_for("load_from_recv_buffers_rank_nminus1",range_1d(1, nx - 1), load_from_recv_buffers_rank_nminus1(fB, fT_recv, fB_recv, nx));
        }
      }

      time_comm += timer2.seconds();
      timer2.reset();

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

        if (params.rank == 0){
          printf("...output step = %i\n", step + 1);
        }

        // deep copy from device to host
        Kokkos::deep_copy(h_u, u);
        Kokkos::deep_copy(h_v, v);
        Kokkos::deep_copy(h_rho, rho);

        // write to output file
        o.write_view("output/u", h_u, params);
        o.write_view("output/v", h_v, params);
        o.write_view("output/rho", h_rho, params);

        o.frame += 1;
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

