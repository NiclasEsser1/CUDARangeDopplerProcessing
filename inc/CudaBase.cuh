#ifndef CUDABASE_H_
#define CUDABASE_H_

#include <cuda.h>
#include <cuda_runtime_api.h>
#include <cufft.h>
#include <libgpujpeg/gpujpeg.h>
#include <nvjpeg.h>
#include <stdio.h>
#include <string>
#include <stdlib.h>
#include <unistd.h>
#include <fstream>  // jpeg delete later
#include <iostream>

#include "CudaGPU.cuh"
#include "CudaVector.cuh"
#include "CudaKernels.cuh"

#define CUDA_CHECK_FFT(ans) {__cufftSafeCall((ans), __FILE__, __LINE__); }

#define MAX_NOF_THREADS 1024
#define TILE_DIM 32			// for transpose functions
#define BLOCK_ROWS 8		// for transpose functions

enum { HAMMING, HANN, BARTLETT, BLACKMAN };
typedef enum { REAL, COMPLEX } numKind;
typedef enum { JET, VIRIDIS, ACCENT, MAGMA, INFERNO, BLUE} color_t;

const int blockSize1 = 2048;
const int threads = 64;

class CudaBase
{
private:
	CudaGPU* device;
public:
	CudaBase(CudaGPU* device);
	~CudaBase();
	void hilbertTransform(float* idata, cufftComplex* odata, int n, int batch);
    void absolute(cufftComplex* idata, float* odata, int width, int height);
    void r2c1dFFT(cufftComplex *odata, int n, int batch, cufftReal* idata = NULL );
    void c2c1dInverseFFT(cufftComplex* idata, int n, int batch);
    void c2c1dFFT(cufftComplex* idata, int n, int batch);
	void mapColors(float* idata, unsigned char* odata, int width, int height, color_t type = JET);
	void hermitianTranspose(cufftComplex* odata, int width, int height, cufftComplex* idata = NULL);
	void fftshift(cufftComplex* data, int n, int batch = 1);
	void encodeBmpToJpeg(unsigned char* idata, uint8_t* odata, int* p_jpeg_size, int width, int height);
	template <typename T> void transpose(T* odata, int width, int height, T* idata = NULL);
	template <typename T> void transposeShared(T* data, int width, int height);
	template <typename T> void window(T* idata, float* window, int width, int height);
	template <typename T> T max(T* idata, int width, int height);
	template <typename T> T min(T* idata, int width, int height);
	void minMax(float* idata, float* odata, int num_els);
	void findBlockSize(int* whichSize, int* num_el);

	void setDevice(CudaGPU* val){device = val;}
    void setWindow(float* idata, int width, int type = HAMMING);

	CudaGPU* getDevice(){return device;}

    void printWindowTaps(float* idata, int win_len);


protected:

	void r2cManyFFT(float* idata, cufftComplex* odata, int* nfft, int rank);
	void c2cManyInPlaceFFT(cufftComplex* data, int *nfft, int rank);
	void c2cManyInPlaceIFFT(cufftComplex* data, int *nfft, int rank);
};


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


#endif
