IDIR1=inc/
IDIR2=/usr/include/
IDIR3=/usr/local/cuda-10.1/include/
LDIR=/usr/lib/
SDIR=src/
ODIR=obj/
BDIR=bin/

CPP=g++
NVCC=nvcc
LIBS=-lstdc++ -lcuda -lcudart -lcufft -lrt -lm

_CSRC =main.cpp CudaGPU.cu CudaBase.cu CudaKernels.cu
CSRC=$(patsubst %,$(SRC)%,$(_CSRC))

_CDEPS=CudaGPU.cuh CudaBase.cuh CudaKernels.cuh CudaVector.cuh
CDEPS=$(patsubst %,$(IDIR1)%,$(_CDEPS))

_OBJ=main.o CudaGPU.o CudaBase.o CudaKernels.o
OBJ=$(patsubst %,$(ODIR)%,$(_OBJ))

$(ODIR)%.o: $(SDIR)%.cpp $(CDEPS)
	$(NVCC) -I$(IDIR1) -I$(IDIR2) -I$(IDIR3) -c $< -o $@

$(ODIR)%.o: $(SDIR)%.cu $(CDEPS)
	$(NVCC) -I$(IDIR1) -I$(IDIR2) -I$(IDIR3) -G -g -O0 --compile --relocatable-device-code=false -gencode arch=compute_61,code=compute_61 -gencode arch=compute_61,code=sm_61  -x cu -o $< -o $@

# $(BDIR)run: $(OBJ)
# 	$(NVCC) --cudart static -L/usr/lib -L/usr/local/cuda-10.1/lib64 --relocatable-device-code=false -gencode arch=compute_61,code=compute_61 -gencode arch=compute_61,code=sm_61 -link -o $(OBJ) $@ $(LIBS)
clean:
	rm -f $(ODIR)*.o $(BDIR)run
