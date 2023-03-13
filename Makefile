# A simple Makefile to build both mpihello w/o cuda and cudahello with cuda calls.
CC=CC
CFLAGS= -O3

#CFLAGS += -DDEBUG

ifeq ($(openacc), y)
	CFLAGS += -acc
endif
ifeq ($(openmp), y)
	CFLAGS += -mp
endif
ifeq ($(openmp_target), y)
	CFLAGS += -mp=gpu -DOPENMP_TARGET
endif

main.ex: main.cpp
	$(CC) $(CFLAGS) -o $@ main.cpp $(LINKFLAGS)

clean:
	rm -f *.exe *.ex *.o

