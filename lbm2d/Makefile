#!/bin/bash

FOLDER=.
KOKKOS_PATH= ../../Kokkos/kokkos
KOKKOS_DEVICES = OpenMP
KOKKOS_USE_TPLS = hwloc
KOKKOS_OPTIONS="disable_deprecated_code"
HWLOC_PATH=/usr/include/hwloc
MPICC?=mpicxx

SRC_PATH = $(FOLDER)/source
LBM_PATH = $(SRC_PATH)/include
RELEASE = $(FOLDER)/release
EXE_NAME = $(FOLDER)/release/lbm

SRC = $(wildcard $(SRC_PATH)/*.cpp)
HDR = $(wildcard *.h)

default: build
#	echo "Start Build"

ifneq (,$(findstring Cuda,$(KOKKOS_DEVICES)))
CXX = ${KOKKOS_PATH}/bin/nvcc_wrapper
EXE = ${EXE_NAME}.cuda
KOKKOS_ARCH = "SNB,Maxwell52"
KOKKOS_CUDA_OPTIONS=enable_lambda
else
CXX = ${MPICC}
EXE = ${EXE_NAME}.host
KOKKOS_ARCH = "SNB"
endif

CXXFLAGS = -Werror -O3 -march=native -mtune=native -DNDEBUG                

INCLUDE = -I$(LBM_PATH)

override CXXFLAGS += -I./ 
LINK = ${CXX}
LINKFLAGS =

DEPFLAGS = -M

vpath %.cpp $(sort $(dir $(SRC)))

OBJ = $(addprefix $(RELEASE)/,$(notdir $(SRC:.cpp=.o)))
LIB =

include $(KOKKOS_PATH)/Makefile.kokkos

$(warning $(OBJ) $(EXE) $(sort $(dir $(SRC))))

build: $(EXE)

$(EXE): $(OBJ) $(KOKKOS_LINK_DEPENDS) 
	$(LINK) $(KOKKOS_LDFLAGS) $(LINKFLAGS) $(EXTRA_PATH) $(OBJ) $(KOKKOS_LIBS) $(LIB) -o $(EXE)

clean: kokkos-clean
	rm -f $(RELEASE)/*.o $(RELEASE)/*.cuda $(RELEASE)/*.host $(RELEASE)/*.a $(RELEASE)/*.h $(RELEASE)/*.tmp $(RELEASE)/*.x $(EXE)

# Compilation rules
$(RELEASE)/%.o:%.cpp $(KOKKOS_CPP_DEPENDS) $(HDR)
	$(CXX) $(KOKKOS_CPPFLAGS) $(KOKKOS_CXXFLAGS) $(CXXFLAGS) $(INCLUDE) -c $< -o $@
