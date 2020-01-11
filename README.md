# lbm2d-mpi-kokkos
## Overview
lbm2d-mpi-kokkos is a C++ 2D weakly compressible fluid flow solver written for distributed CPUs or GPUs. The app is built on top of Sandia National Lab's performance portable Kokkos library. The majority of the app was written while attending a Kokkos workshop hosted by ORNL in July of 2018. Boundary conditions are specific to the 2D lid driven cavity. If you want something else, feel free to modify/extend the code.

## Build
```bash
mkdir proj
cd proj
git clone https://github.com/vasylivy/lbm2d-mpi-kokkos.git
```
Edit `lbmd2d-mpi-kokkos/lbm2d/Makefile` and set the device, architecture, and path for Kokkos. For example, setting `KOKKOS_DEVICES = OpenMP` and `KOKKOS_ARCH = SNB` will compile for Sandy Bridge Intel CPUs + OpenMP. Setting `KOKKOS_DEVICES = Cuda` and `KOKKOS_ARCH = SNB, Maxwell52`, will compile for a Sandy Bridge Intel CPU +  Nvidia GTX Titan X. A list of architecture flags, as well as other Kokkos compilation flags can be found [here](https://github.com/kokkos/kokkos/wiki/Compiling). The default Kokkos path is `KOKKOS_PATH = ../../Kokkos/kokkos`. While still in the proj directory, execute the following cmds to download Kokkos

```bash
mkdir Kokkos
cd Kokkos
git clone https://github.com/kokkos/kokkos.git
```
To build the executable

```bash
cd ../lbm2d-mpi-kokkos/lbm2d
mkdir release
make -j
```
The app will be compiled to the `release` directory as `lbm.host` or `lbm.cuda` depending on if it is an OpenMP or Cuda build.

## lattice Boltzmann method
The LBM implementation uses the BGK collision operator, D2Q9 quadrature for velocity space, and half-way bounce back for the boundary conditions. For collision-streaming, an A-B pattern is used. That is 9 distriubtions are read from array A, collided, and then streamed to array B. After applying boundary conditions, pointers to the arrays are then swapped. This collision-streaming pattern requires 2x the memory storage but is easy to implement. Details and further references regarding LBM can be found in Kruger's et al. text [[1]](https://link.springer.com/content/pdf/10.1007/978-3-319-44649-3.pdf).

## Running the 2D lid driven cavity
After compilation, execute the cmds

```bash
cd lbm2d-mpi-kokkos/cavity2d
./run.sh
```
The app outputs macroscopic variables to `cavity2d/output/` as `.bin` files based on the specified output rate until steady state is reached or the number of maximum steps is exceeded.

Results can be plotted and compared to the Ghia data set using `plot.py` as follows

```bash
python3 ../scripts/plot.py -reynolds 100 400 1000 5000 -nx 512 -ny 512 -ulid 0.1
```
The python scripts assumes the `output` for a desired Reynolds number has been moved to its own directory. For `Re=100` with a domain size of `512x512`, the output should be moved to a directory `re100_512x512`. By default, a row major (C/C++) data format is assumed. An optional `-cuda 1` flag can be passed to specify to read the `*.bin` using column-major (Cuda/Fortran) data format.

## Validation

The lid driven cavity is an enclosed flow driven by the movement of a lid. Below a critical Reynolds number, the flow is steady. The figures below compare the horizontal and vertical component of velocity at the vertical centerline and horizontal centerline for different Reynolds numbers (`Re = 100, 400, 1000, 5000`) against Ghia et al. (1982) results. The LBM simulations use a domain size of `512x512` corresponding to a spatial resolution of `510x510`.

<p align="center">
  <img src="cavity2d/512x512_u.png" height="275">
  <img src="cavity2d/512x512_v.png" height="275">
</p>

## Colorful Fluid Dynamics 

Below are animations of the velocity magnitude (left) and vorticity (right) for the 2D lid driven cavity over the first 25 million steps. This unsteady flow corresponds to a Reynolds number of `Re=25000` resolved on `2048x2048` domain with a lid speed set to `ulid = 0.01` which results in a Mach number `Ma ~ 0.017`. A single Nvidia GTX Titan X was used with a throughput of approximately ~1.3 BLUPs (Billion Lattice Updates Per Second).

<p align="center">
  <img src="cavity2d/mag.gif" >
  <img src="cavity2d/vorticity.gif">
</p>

The python script `scripts/dump_frames.py` can be called to dump images of the vorticity or velocity magnitude to the current directory.

```bash
python3 ../../scripts/animate.py -re 25000 -nx 2048 -ny 2048 -umax 0.01 -cuda 1 -curl 0
```

Use the `ffmpeg` library to generate the GIF animation

```bash
ffmpeg -v 0 -i img_%03d.png -vf palettegen -y palette.png
ffmpeg -v 0 -framerate 10 -loop 0 -i img_%03d.png -i palette.png -lavfi paletteuse -y out.gif
```

## Performance 

### Kokkos profiling

### Shared Memory

### Distributed Memory


