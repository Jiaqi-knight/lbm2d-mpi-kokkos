#!/bin/bash

[ ! -d "output" ] && mkdir output
app=../lbm2d/release/lbm.host
nx=512;
ny=512;
re=400;
tol=.00001;
max_steps=50000;
output_rate=50000;

# pure OpenMP
np=1;
nt=32;
#export GOMP_CPU_AFFINITY="0 8 1 9 2 10 3 11 4 12 5 13 6 14 7 15 16 24 17 25 18 26 19 27 20 28 21 29 22 30 23 31"
#export OMP_DISPLAY_ENV=true


#export OMP_NUM_THREADS=32
mpirun -N $np -bind-to none -x OMP_NUM_THREADS=$nt -x OMP_PROC_BIND=spread -x OMP_PLACES=threads $app $nx $ny $re $tol $max_steps $output_rate

# hybrid MPI/OpenMP
#np=16;
#nt=2;
#mpirun -N $np -bind-to none -x OMP_NUM_THREADS=$nt $app $nx $ny $re $tol $max_steps $output_rate
