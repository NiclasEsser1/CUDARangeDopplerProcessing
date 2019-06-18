IDIR1=inc/
IDIR2=/usr/include/
LDIR=/usr/lib/
SDIR=src/
ODIR=obj/
BDIR=bin/

CPP=g++
NVCC=nvcc
LIBS=-lstdc++ -lcudart -lcufft -lrt -lm

_CSRC =main.cpp SignalGenerator.cpp Stopwatch.cpp CudaGPU.cu CudaBase.cu CudaKernels.cu
CSRC=$(patsubst %,$(SRC)%,$(_CSRC))

_CDEPS=main.h SignalGenerator.h Stopwatch.h CudaGPU.cuh CudaBase.cuh CudaKernels.cuh CudaVector.cuh
CDEPS=$(patsubst %,$(IDIR1)%,$(_CDEPS))

_OBJ=main.o SignalGenerator.o Stopwatch.o CudaGPU.o CudaBase.o CudaKernels.o CudaVector.o
OBJ=$(patsubst %,$(ODIR)%,$(_OBJ))

$(ODIR)%.o: $(SDIR)%.cpp $(CDEPS)
	$(NVCC) -I$(IDIR1) -I$(IDIR2) -c $< -o $@

$(ODIR)%.o: $(SDIR)%.cu $(CDEPS)
	$(NVCC) -I$(IDIR1) -I$(IDIR2) -c $< -o $@

$(BDIR)run: $(OBJ) $(CRSC)
	$(NVCC) $(OBJ) -o $@ $(LIBS)
clean:
	rm -f $(ODIR)*.o $(BDIR)run