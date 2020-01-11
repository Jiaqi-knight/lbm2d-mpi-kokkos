#ifndef _PARAMS_H

#define _PARAMS_H

#include "typedefs.h"

struct Params {

  int nx = 100;
  int ny = 100;
  int max_steps = 1000;
  int output_rate = 100;

  int num_proc, rank;

  Double re = 100.;
  Double tau, u_lid, nu, tol;
};

#endif
