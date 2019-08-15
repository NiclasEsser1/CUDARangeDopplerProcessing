IDIR1=inc/
IDIR2=/usr/include/
IDIR3=/usr/local/cuda-10.1/include/
LDIR1=/usr/lib/
LDIR2=/usr/lib/x86_64-linux-gnu
SDIR=src/
ODIR=obj/
BDIR=bin/

CPP=g++
NVCC=nvcc

LIBS=-lstdc++ -lcuda -lcudart -lcufft -lrt -lm

_CSRC =main.cpp SignalGenerator.cpp CudaGPU.cu CudaBase.cu CudaAlgorithm.cu CudaKernels.cu
CSRC=$(patsubst %,$(SRC)%,$(_CSRC))

_CDEPS=SignalGenerator.h Bitmap_IO.h CudaTest.h CudaGPU.cuh CudaBase.cuh CudaAlgorithm.cuh CudaKernels.cuh CudaVector.cuh
CDEPS=$(patsubst %,$(IDIR1)%,$(_CDEPS))

_OBJ=main.o SignalGenerator.o CudaGPU.o CudaBase.o CudaAlgorithm.o CudaKernels.o
OBJ=$(patsubst %,$(ODIR)%,$(_OBJ))

$(ODIR)%.o: $(SDIR)%.cpp $(CDEPS)
	$(NVCC) -I$(IDIR1) -I$(IDIR2) -I$(IDIR3) -c $< -o $@

$(ODIR)%.o: $(SDIR)%.cu $(CDEPS)
	$(NVCC) -I$(IDIR1) -I$(IDIR2) -I$(IDIR3) -c $< -o $@

$(BDIR)run: $(OBJ)
	$(NVCC) -L$(LDIR1) -L$(LDIR2) $(OBJ) -o $@ $(LIBS)
clean:
	rm -f $(ODIR)*.o $(BDIR)run
