#ifndef CUDABASE_H_
#define CUDABASE_H_

#include <cuda_runtime_api.h>
#include <cufft.h>
#include <stdio.h>
#include <stdlib.h>     /* exit, EXIT_FAILURE */

#include "CudaGPU.cuh"
#include "CudaVector.cuh"

#define MAX_NOF_THREADS 1024
#define TRANSPOSE_DIM 32
#define TRANSPOSE_ROWS 32

#define CUDA_CHECK_FFT(ans) {__cufftCechError((ans), __FILE__, __LINE__); }

inline void __cufftCechError(cufftResult code, const char *file, const int line, bool abort = true)
{
    if( CUFFT_SUCCESS != code) {
        fprintf(stderr, "cufftSafeCall() CUFFT error in file <%s>, line %i.\n",
            file,
			line);
        if (abort) exit(code);
    }
}



typedef enum { HAMMING, HANN, BARTLETT, BLACKMAN } winType;
typedef enum { REAL, COMPLEX } numKind;
typedef enum { HOST, D_GLOBAL, D_SHARED } winLocation;

typedef struct
{
	int x;
}med_filter;


class CudaBase
{
public:
	CudaBase(CudaGPU* device, int length, int height = 1, int depth = 1);
	~CudaBase();
	void initDeviceEnv();
	void rangeDopplerProcessing();
	void hilbertTransform(float* idata, cufftComplex* odata);
	void doFFT(cufftComplex* idata, cufftComplex* odata = NULL, bool transpose = false);
	void windowCplx(cufftComplex* idata);
	void make_medianfilter();
	void setLength(int val);
	void setHeight(int val);
	void setDepth(int val);
	void setBytes(int val);
	void setMedFilter(int val);
	void setDevice(CudaGPU* val);
	void setWindow(winType type = HAMMING, numKind kind = REAL, winLocation loc = D_GLOBAL);
    void printWindowTaps();
    template<typename T> void window(T* idata);
	void setProcessingBuffer(float* buf, cudaMemcpyKind kind = cudaMemcpyHostToDevice);

private:
	CudaGPU* device;
	CudaVector<float>* input;
	CudaVector<float>* output;
	CudaVector<float>* windowBuf;
	CudaVector<float>* filterBuf;
	CudaVector<cufftComplex>* fftBuf;
	CudaVector<cufftComplex>* rngDopBuf;
    winType win_type;
	numKind win_kind;
	winLocation win_loc;
	int win_len;
	med_filter m_filter;
	int x;
	int y;
	int z;
	int bytes;

protected:
    template<typename T> void memCopyBuffer(T* dst, T* src, size_t size, cudaMemcpyKind kind = cudaMemcpyHostToDevice);
    template<typename T> void allocateDeviceMem(T* buf, size_t elements);
    template<typename T> void freeCudaVector(CudaVector<T*>* vec);
	void freeDeviceMem(void* ptr);
	void calculateWindowTaps();
	void r2cManyFFT(float* idata, cufftComplex* odata, int* nfft, int rank);
	void c2cManyInPlaceFFT(cufftComplex* data, int *nfft, int rank);
	void c2cManyInPlaceIFFT(cufftComplex* data, int *nfft, int rank);
};

#endif
