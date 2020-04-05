IDIR1=inc/
IDIR2=/usr/include/
IDIR3=/usr/local/cuda/include/
LDIR1=/usr/lib/
LDIR2=/usr/lib/x86_64-linux-gnu
LDIR3=/usr/local/lib
SDIR=src/
ODIR=obj/
BDIR=bin/

CPP=g++
NVCC=nvcc

LIBS=-lstdc++ -lcuda -lcudart -lcurand -lcufft -lnvjpeg -lrt -lm -lgpujpeg

_CDEPS=signalgenerator.h bitmap_io.h cudatest.h cudagpu.cuh cudabase.cuh cudaalgorithm.cuh cudakernels.cuh cudavector.cuh utils.h
CDEPS=$(patsubst %,$(IDIR1)%,$(_CDEPS))

_OBJ1=test_cudaprocessing.o signalgenerator.o cudagpu.o cudabase.o cudaalgorithm.o cudakernels.o utils.o
OBJ1=$(patsubst %,$(ODIR)%,$(_OBJ1))

_OBJ2=benchmark_kernels.o signalgenerator.o cudagpu.o cudabase.o cudaalgorithm.o cudakernels.o utils.o
OBJ2=$(patsubst %,$(ODIR)%,$(_OBJ2))

_OBJ4=offline_processing.o signalgenerator.o cudagpu.o cudabase.o cudaalgorithm.o cudakernels.o utils.o
OBJ4=$(patsubst %,$(ODIR)%,$(_OBJ4))

_OBJ5=benchmark_memory_bandwidth.o cudagpu.o cudabase.o cudaalgorithm.o cudakernels.o utils.o
OBJ5=$(patsubst %,$(ODIR)%,$(_OBJ5))

_OBJ6=benchmark_fft.o cudagpu.o cudabase.o cudaalgorithm.o cudakernels.o utils.o
OBJ6=$(patsubst %,$(ODIR)%,$(_OBJ6))

_OBJ7=benchmark_algorithm.o cudagpu.o cudabase.o cudaalgorithm.o cudakernels.o utils.o
OBJ7=$(patsubst %,$(ODIR)%,$(_OBJ7))

$(ODIR)%.o: $(SDIR)%.cpp $(CDEPS)
	$(NVCC) -I$(IDIR1) -I$(IDIR2) -I$(IDIR3) -c $< -o $@

$(ODIR)%.o: $(SDIR)%.cu $(CDEPS)
	$(NVCC) -I$(IDIR1) -I$(IDIR2) -I$(IDIR3) -c $< -o $@

all: $(BDIR)test_cudaprocessing $(BDIR)benchmark_kernels $(BDIR)benchmark_fft $(BDIR)benchmark_algorithm $(BDIR)benchmark_memory_bandwidth $(BDIR)offline_processing

$(BDIR)test_cudaprocessing: $(OBJ1)
	$(NVCC) -L$(LDIR1) -L$(LDIR2) -L$(LDIR3) $(OBJ1) -o $@ $(LIBS)

$(BDIR)benchmark_kernels: $(OBJ2)
	$(NVCC) -L$(LDIR1) -L$(LDIR2) -L$(LDIR3) $(OBJ2) -o $@ $(LIBS)

$(BDIR)benchmark_memory_bandwidth: $(OBJ5)
	$(NVCC) -L$(LDIR1) -L$(LDIR2) -L$(LDIR3) $(OBJ5) -o $@ $(LIBS)

$(BDIR)benchmark_fft: $(OBJ6)
	$(NVCC) -L$(LDIR1) -L$(LDIR2) -L$(LDIR3) $(OBJ6) -o $@ $(LIBS)

$(BDIR)benchmark_algorithm: $(OBJ7)
	$(NVCC) -L$(LDIR1) -L$(LDIR2) -L$(LDIR3) $(OBJ7) -o $@ $(LIBS)

$(BDIR)offline_processing: $(OBJ4)
	$(NVCC) -L$(LDIR1) -L$(LDIR2) -L$(LDIR3) $(OBJ4) -o $@ $(LIBS)

clean:
	rm -f $(ODIR)*.o $(BDIR)test_cudaprocessing $(BDIR)benchmark_kernels $(BDIR)benchmark_memory_bandwidth $(BDIR)benchmark_fft $(BDIR)benchmark_algorithm $(BDIR)offline_processing
