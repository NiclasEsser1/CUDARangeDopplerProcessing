IDIR1=inc/
IDIR2=/usr/include/
IDIR3=/usr/local/cuda-10.1/include/
LDIR1=/usr/lib/
LDIR2=/usr/lib/x86_64-linux-gnu
LDIR3=/usr/local/lib
SDIR=src/
ODIR=obj/
BDIR=bin/

CPP=g++
NVCC=nvcc

LIBS=-lstdc++ -lcuda -lcudart -lcufft -lnvjpeg -lrt -lm -lgpujpeg

_CDEPS=SignalGenerator.h Bitmap_IO.h Socket.h TCPConfig.h CudaTest.h CudaGPU.cuh CudaBase.cuh CudaAlgorithm.cuh CudaKernels.cuh CudaVector.cuh
CDEPS=$(patsubst %,$(IDIR1)%,$(_CDEPS))

_OBJ1=test.o SignalGenerator.o CudaGPU.o CudaBase.o CudaAlgorithm.o CudaKernels.o
OBJ1=$(patsubst %,$(ODIR)%,$(_OBJ1))

_OBJ2=benchmark.o SignalGenerator.o CudaGPU.o CudaBase.o CudaAlgorithm.o CudaKernels.o
OBJ2=$(patsubst %,$(ODIR)%,$(_OBJ2))

_OBJ3=streaming_example.o SignalGenerator.o Socket.o CudaGPU.o CudaBase.o CudaAlgorithm.o CudaKernels.o
OBJ3=$(patsubst %,$(ODIR)%,$(_OBJ3))




$(ODIR)%.o: $(SDIR)%.cpp $(CDEPS)
	$(NVCC) -I$(IDIR1) -I$(IDIR2) -I$(IDIR3) -c $< -o $@

$(ODIR)%.o: $(SDIR)%.cu $(CDEPS)
	$(NVCC) -I$(IDIR1) -I$(IDIR2) -I$(IDIR3) -c $< -o $@

all: $(BDIR)test $(BDIR)benchmark $(BDIR)streaming_example

$(BDIR)test: $(OBJ1)
	$(NVCC) -L$(LDIR1) -L$(LDIR2) -L$(LDIR3) $(OBJ1) -o $@ $(LIBS)

$(BDIR)benchmark: $(OBJ2)
	$(NVCC) -L$(LDIR1) -L$(LDIR2) -L$(LDIR3) $(OBJ2) -o $@ $(LIBS)

$(BDIR)streaming_example: $(OBJ3)
	$(NVCC) -L$(LDIR1) -L$(LDIR2) -L$(LDIR3) $(OBJ3) -o $@ $(LIBS)


clean:
	rm -f $(ODIR)*.o $(BDIR)test $(BDIR)benchmark $(BDIR)streaming_example
