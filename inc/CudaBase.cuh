#ifndef CUDABASE_H_
#define CUDABASE_H_

#include <cuda.h>
#include <cuda_runtime_api.h>
#include <cufft.h>
#include <stdio.h>
#include <stdlib.h>     /* exit, EXIT_FAILURE */
#include <string>

#include "CudaGPU.cuh"
#include "CudaVector.cuh"

#define MAX_NOF_THREADS 1024
#define TRANSPOSE_DIM 32
#define TRANSPOSE_ROWS 32

#define CUDA_CHECK_FFT(ans) {__cufftSafeCall((ans), __FILE__, __LINE__); }

inline void __cufftSafeCall(cufftResult code, const char *file, const int line, bool abort = true)
{
    std::string err_str;
    if( CUFFT_SUCCESS != code)
    {
        switch (code)
        {
            case CUFFT_INVALID_PLAN:
                err_str = "CUFFT_INVALID_PLAN";

            case CUFFT_ALLOC_FAILED:
                err_str = "CUFFT_ALLOC_FAILED";

            case CUFFT_INVALID_TYPE:
                err_str = "CUFFT_INVALID_TYPE";

            case CUFFT_INVALID_VALUE:
                err_str = "CUFFT_INVALID_VALUE";

            case CUFFT_INTERNAL_ERROR:
                err_str = "CUFFT_INTERNAL_ERROR";

            case CUFFT_EXEC_FAILED:
                err_str = "CUFFT_EXEC_FAILED";

            case CUFFT_SETUP_FAILED:
                err_str = "CUFFT_SETUP_FAILED";

            case CUFFT_INVALID_SIZE:
                err_str = "CUFFT_INVALID_SIZE";

            case CUFFT_UNALIGNED_DATA:
                err_str = "CUFFT_UNALIGNED_DATA";
        }
        printf("\ncufftSafeCall() Error: %s in file <%s>, line %i.\n",
            err_str.c_str(),
            file,
			line);
        if (abort) exit(code);
    }
}



typedef enum { HAMMING, HANN, BARTLETT, BLACKMAN } winType;
typedef enum { REAL, COMPLEX } numKind;

typedef struct
{
	int x;
}med_filter;


class CudaBase
{
public:
	CudaBase(CudaGPU* device);
	~CudaBase();
	void initDeviceEnv();
    void freeMemory();
	void hilbertTransform(float* idata, cufftComplex* odata, int n, int batch);
    void absolute(cufftComplex* idata, float* odata, int width, int height);
    void transpose(cufftComplex* idata, cufftComplex* odata, int width, int height);
    void r2c1dFFT(float* idata, cufftComplex *odata, int n, int batch);
    void c2c1dIFFT(cufftComplex* idata, int n, int batch);
    void c2c1dFFT(cufftComplex* idata, int n, int batch);
    void windowReal(float* idata, int width, int height);
    void windowCplx(cufftComplex* idata, int width, int height);

    void setLength(int val){x_size = val;}
	void setHeight(int val){y_size = val;}
	void setDevice(CudaGPU* val){device = val;}
    void setWindow(winType type = HAMMING, numKind kind = REAL);
	float* getFloatBuffer(){return floatBuffer->getDevPtr();}
	cufftComplex* getComplexBuffer(){return complexBuffer->getDevPtr();}
	cufftComplex* getTransposeBuffer(){return transposeBuf->getDevPtr();}

    void printWindowTaps();
private:
	CudaGPU* device;
	CudaVector<float>* floatBuffer;
	CudaVector<float>* windowBuf;
	CudaVector<cufftComplex>* complexBuffer;
	CudaVector<cufftComplex>* transposeBuf;
    winType win_type;
	numKind win_kind;
	int win_len;
	int x_size;
	int y_size;
protected:
    template<typename T> void memCopyBuffer(T* dst, T* src, size_t size, cudaMemcpyKind kind = cudaMemcpyHostToDevice);
    template<typename T> void allocateDeviceMem(T* buf, size_t elements);
    template<typename T> void freeCudaVector(CudaVector<T>* vec)
    {
    	if (vec != NULL)
    	{
    		vec->~CudaVector();
    	}
    }
	void freeDeviceMem(void* ptr);
	void calculateWindowTaps();
	void r2cManyFFT(float* idata, cufftComplex* odata, int* nfft, int rank);
	void c2cManyInPlaceFFT(cufftComplex* data, int *nfft, int rank);
	void c2cManyInPlaceIFFT(cufftComplex* data, int *nfft, int rank);
};

#endif
