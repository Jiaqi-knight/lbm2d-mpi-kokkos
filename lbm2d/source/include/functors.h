#include <typedefs.h>
#include <params.h>
#include <math.h>
#include <sstream>

template <typename T>
KOKKOS_INLINE_FUNCTION T max(T x1, T x2) {
  return (x1 > x2) ? x1 : x2;
}

struct collide_stream {

  const Params params;
  const Double tau = params.tau;
  const Double tau_inv = 1. / params.tau;
  const Double wo = 4. / 9.;
  const Double ws = 1. / 9;
  const Double wd = 1. / 36;
  const Double omtau_inv = (1.0 - tau_inv);
  const Double ulid = params.u_lid;

  const DistributionField fB;
  const DistributionField fA;

  collide_stream(DistributionField fB, DistributionField fA, Params params) :
      fB(fB), fA(fA), params(params) {
  }

  KOKKOS_INLINE_FUNCTION
  void operator()(const int i, const int j) const {
    // load distributions
    Double f0 = fA(i, j, 0);
    Double f1 = fA(i, j, 1);
    Double f2 = fA(i, j, 2);
    Double f3 = fA(i, j, 3);
    Double f4 = fA(i, j, 4);
    Double f5 = fA(i, j, 5);
    Double f6 = fA(i, j, 6);
    Double f7 = fA(i, j, 7);
    Double f8 = fA(i, j, 8);

    // compute density
    Double density = f0 + f1 + f2 + f3 + f4 + f5 + f6 + f7 + f8;
    Double density_inv = 1. / density;

    // compute velocities
    Double ux = (f1 + f5 + f8 - (f3 + f6 + f7)) * density_inv;
    Double uy = (f2 + f5 + f6 - (f4 + f7 + f8)) * density_inv;

    // update distribubtions
    Double tw0r = wo * density * tau_inv; // w[0]*rho
    Double twsr = ws * density * tau_inv; // w[1-4]*rho
    Double twdr = wd * density * tau_inv; // w[5-8]*rho

    Double tu = 3.0 * ux;
    Double tv = 3.0 * uy;

    Double a = 1.0 - 1.5 * (ux * ux + uy * uy);
    Double udot5 = tu + tv;
    Double udot6 = -tu + tv;

    Double feq0 = tw0r * a;
    Double feq1 = twsr * (a + tu * (1.0 + 0.5 * tu));
    Double feq3 = twsr * (a - tu * (1.0 - 0.5 * tu));
    Double feq2 = twsr * (a + tv * (1.0 + 0.5 * tv));
    Double feq4 = twsr * (a - tv * (1.0 - 0.5 * tv));
    Double feq5 = twdr * (a + udot5 * (1.0 + 0.5 * udot5));
    Double feq7 = twdr * (a - udot5 * (1.0 - 0.5 * udot5));
    Double feq6 = twdr * (a + udot6 * (1.0 + 0.5 * udot6));
    Double feq8 = twdr * (a - udot6 * (1.0 - 0.5 * udot6));

    Double fB0 = omtau_inv * f0 + feq0;
    Double fB1 = omtau_inv * f1 + feq1;
    Double fB2 = omtau_inv * f2 + feq2;
    Double fB3 = omtau_inv * f3 + feq3;
    Double fB4 = omtau_inv * f4 + feq4;
    Double fB5 = omtau_inv * f5 + feq5;
    Double fB6 = omtau_inv * f6 + feq6;
    Double fB7 = omtau_inv * f7 + feq7;
    Double fB8 = omtau_inv * f8 + feq8;

    // stream distributions
    fB(i, j, 0) = fB0;
    fB(i, j + 1, 1) = fB1;
    fB(i + 1, j, 2) = fB2;
    fB(i, j - 1, 3) = fB3;
    fB(i - 1, j, 4) = fB4;
    fB(i + 1, j + 1, 5) = fB5;
    fB(i + 1, j - 1, 6) = fB6;
    fB(i - 1, j - 1, 7) = fB7;
    fB(i - 1, j + 1, 8) = fB8;
  }
};

struct bb_west {

  const int j = 1;
  const DistributionField fB;

  bb_west(DistributionField fB) :
      fB(fB) {
  }

  KOKKOS_INLINE_FUNCTION
  void operator()(const int i) const {

    fB(i, j, 1) = fB(i, j - 1, 3);
    fB(i, j, 5) = fB(i - 1, j - 1, 7);
    fB(i, j, 8) = fB(i + 1, j - 1, 6);

  }
};

struct bb_east {

  const int j;
  const DistributionField fB;

  bb_east(DistributionField fB) :
      fB(fB), j(fB.extent(1) - 2) {
  }

  KOKKOS_INLINE_FUNCTION
  void operator()(const int i) const {
    fB(i, j, 3) = fB(i, j + 1, 1);
    fB(i, j, 6) = fB(i - 1, j + 1, 8);
    fB(i, j, 7) = fB(i + 1, j + 1, 5);
  }
};

struct bb_south {

  const int i = 1;
  const DistributionField fB;

  bb_south(DistributionField fB) :
      fB(fB) {
  }

  KOKKOS_INLINE_FUNCTION
  void operator()(const int j) const {

    fB(i, j, 2) = fB(i - 1, j, 4);
    fB(i, j, 6) = fB(i - 1, j + 1, 8);
    fB(i, j, 5) = fB(i - 1, j - 1, 7);

  }
};

struct bb_north {

  const Double wd = 1. / 36;
  const Double ulid;
  const int i;
  const DistributionField fB;

  bb_north(DistributionField fB, Double ulid) :
      fB(fB), ulid(ulid), i(fB.extent(0) - 2) {
  }

  KOKKOS_INLINE_FUNCTION
  void operator()(const int j) const {

    Double rhs_5 = 6.0 * wd * ulid;
    Double rhs_6 = -6.0 * wd * ulid;

    fB(i, j, 4) = fB(i + 1, j, 2);
    fB(i, j, 7) = fB(i + 1, j + 1, 5) - rhs_5;
    fB(i, j, 8) = fB(i + 1, j - 1, 6) - rhs_6;

  }
};

struct load_from_recv_buffers_rank_zero {

  const int ny;
  const int nx;

  const DistributionField fB;
  const HaloDistField fT_recv;
  const HaloDistField fB_recv;

  load_from_recv_buffers_rank_zero(DistributionField fB, HaloDistField fT_recv, HaloDistField fB_recv, const int ny, const int nx) :
    fB(fB), fT_recv(fT_recv), fB_recv(fB_recv), ny(ny), nx(nx){
  }

  KOKKOS_INLINE_FUNCTION
  void operator()(const int j) const {
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
  }
};

struct load_from_recv_buffers_rank_nminus1 {

  const int nx;
  const DistributionField fB;
  const HaloDistField fT_recv;
  const HaloDistField fB_recv;

  load_from_recv_buffers_rank_nminus1(DistributionField fB, HaloDistField fT_recv, HaloDistField fB_recv, const int nx):
    fB(fB), fT_recv(fT_recv), fB_recv(fB_recv), nx(nx){
  }

  KOKKOS_INLINE_FUNCTION
  void operator()(const int j) const {
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
};

struct load_from_recv_buffers_rank_k{

  const int ny;
  const int nx;

  const DistributionField fB;
  const HaloDistField fT_recv;
  const HaloDistField fB_recv;

  load_from_recv_buffers_rank_k(DistributionField fB, HaloDistField fT_recv, HaloDistField fB_recv, const int ny, const int nx) :
    fB(fB), fT_recv(fT_recv), fB_recv(fB_recv), ny(ny), nx(nx){
  }

  KOKKOS_INLINE_FUNCTION
  void operator()(const int j) const {
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
  }
};

struct populate_send_buffers {

  const int ny;
  const DistributionField fB;
  const HaloDistField fT_send;
  const HaloDistField fB_send;

  populate_send_buffers(DistributionField fB, HaloDistField fT_send, HaloDistField fB_send, const int ny) :
      fB(fB), fT_send(fT_send), fB_send(fB_send), ny(ny) {
  }

  KOKKOS_INLINE_FUNCTION
  void operator()(const int j) const {
    fT_send(j, 0) = fB(ny - 1, j, 5);
    fT_send(j, 1) = fB(ny - 1, j, 2);
    fT_send(j, 2) = fB(ny - 1, j, 6);

    fB_send(j, 0) = fB(0, j, 7);
    fB_send(j, 1) = fB(0, j, 4);
    fB_send(j, 2) = fB(0, j, 8);
  }
};

struct compute_macroscopic {

  const ScalarField u, v, rho;
  const DistributionField f;

  compute_macroscopic(DistributionField f, ScalarField u, ScalarField v, ScalarField rho) :
      f(f), u(u), v(v), rho(rho) {
  }

  KOKKOS_INLINE_FUNCTION
  void operator()(const int i, const int j) const {

    // distributions
    Double f0 = f(i, j, 0);
    Double f1 = f(i, j, 1);
    Double f2 = f(i, j, 2);
    Double f3 = f(i, j, 3);
    Double f4 = f(i, j, 4);
    Double f5 = f(i, j, 5);
    Double f6 = f(i, j, 6);
    Double f7 = f(i, j, 7);
    Double f8 = f(i, j, 8);

    // compute density
    Double density = f0 + f1 + f2 + f3 + f4 + f5 + f6 + f7 + f8;

    // compute velocities
    Double utmp = (f1 + f5 + f8 - (f3 + f6 + f7)) / density;
    Double vtmp = (f2 + f5 + f6 - (f4 + f7 + f8)) / density;

    // write to device
    rho(i, j) = density;
    u(i, j) = utmp;
    v(i, j) = vtmp;

  }
};

struct load_state {

  const ScalarField u, v, rho;
  const DistributionField fA, fB;

  const Double w[9] = { 4. / 9, 1. / 9, 1. / 9, 1. / 9, 1. / 9, 1. / 36, 1. / 36, 1. / 36, 1. / 36 };

  load_state(DistributionField fA, DistributionField fB, ScalarField u, ScalarField v, ScalarField rho) :
      fA(fA), fB(fB), u(u), v(v), rho(rho) {
  }

  KOKKOS_INLINE_FUNCTION
  void operator()(const int i, const int j) const {

    rho(i, j) = 1.0;
    u(i, j) = 0.0;
    v(i, j) = 0.0;

    for (int k = 0; k < fA.extent(2); ++k) {

      Double tmp = w[k] * 1.0;
      fA(i, j, k) = tmp;
      fB(i, j, k) = tmp;
    }
  }
};

struct is_steady_state {

  const Double tol;
  const DistributionField fA;
  const ScalarField u, v, rho;

  is_steady_state(DistributionField fA, ScalarField u, ScalarField v, ScalarField rho, const Double tol) :
      fA(fA), u(u), v(v), rho(rho), tol(tol) {
  }

  KOKKOS_INLINE_FUNCTION
  void operator()(const int i, const int j, int &converged) const {

    // load distributions A
    Double f0 = fA(i, j, 0);
    Double f1 = fA(i, j, 1);
    Double f2 = fA(i, j, 2);
    Double f3 = fA(i, j, 3);
    Double f4 = fA(i, j, 4);
    Double f5 = fA(i, j, 5);
    Double f6 = fA(i, j, 6);
    Double f7 = fA(i, j, 7);
    Double f8 = fA(i, j, 8);

    // compute momentum A
    Double umom_A = (f1 + f5 + f8 - (f3 + f6 + f7));
    Double vmom_A = (f2 + f5 + f6 - (f4 + f7 + f8));

    // load macroscopic properties B
    Double rho_B = rho(i, j);
    Double umom_B = u(i, j) * rho_B;
    Double vmom_B = v(i, j) * rho_B;

    // convergence criteria
    int converged_momu = fabs(umom_B - umom_A) < max(tol * fabs(umom_A), 1e-6);
    int converged_momv = fabs(vmom_B - vmom_A) < max(tol * fabs(vmom_A), 1e-6);

//    if (!converged_momu or !converged_momv){
//      std::stringstream ss; ss << i << "," << j << "\n";
//      std::cout << ss.str();
//    }
    converged &= converged_momu & converged_momv;
  }
};

