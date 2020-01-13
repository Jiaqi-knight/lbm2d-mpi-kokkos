#ifndef _TYPEDEF_H_
#define _TYPEDEF_H_

#include <Kokkos_Core.hpp>


typedef double Double;

typedef Kokkos::View<Double*[3]> HaloDistField;
typedef Kokkos::View<Double**> ScalarField;
typedef Kokkos::View<Double**[9]> DistributionField;
typedef Kokkos::MDRangePolicy< Kokkos::Rank<2>> range_2d;
typedef Kokkos::RangePolicy<> range_1d;

#endif

