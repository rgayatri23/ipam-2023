# A simple Makefile to build both mpihello w/o cuda and cudahello with cuda calls.
CC=clang++
CFLAGS= -O3

CFLAGS += -DDEBUG

ifeq ($(cuda), y)
	CC=nvcc -ccbin CC -m64 --x cu
	CFLAGS += -DCUDA
	CFLAGS += -I/global/cfs/cdirs/m1759/rgayatri/cuda-samples/Common/
endif
ifeq ($(openacc), y)
	CFLAGS += -acc
endif
ifeq ($(openmp), y)
#	CFLAGS += -mp
	CFLAGS += -fopenmp
endif
ifeq ($(openmp_target), y)
#	CFLAGS += -mp=gpu -DOPENMP_TARGET
	CFLAGS +=  --offload-arch=sm_80 -DOPENMP_TARGET
	CFLAGS += -I$(CRAY_MPICH_PREFIX)/include
	LDFLAGS += -L$(CRAY_MPICH_PREFIX)/lib -lmpi
	LDFLAGS += $(PE_MPICH_GTL_DIR_nvidia80) $(PE_MPICH_GTL_LIBS_nvidia80)
endif

main.ex: main.cpp 
	$(CC) $(CFLAGS) $(LDFLAGS) -o $@ main.cpp $(LINKFLAGS)

clean:
	rm -f *.exe *.ex *.o

