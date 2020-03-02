#ifndef CUDAALGORITHM_H_
#define CUDAALGORITHM_H_

#include <iostream>
#include <cuda_runtime.h>
#include <stdio.h>
#include <unistd.h>

#include "CudaGPU.cuh"
#include "CudaBase.cuh"
#include "CudaVector.cuh"
#include "Socket.h"

enum {BMP24, JPEG};

class CudaAlgorithm
{

private:
    CudaGPU* device;
    CudaBase* base;
    int x_size;
    int y_size;
    int z_size;
    int image_size; // Number of bytes of image
    int color_depth;
    double total_required_mem;
	bool allocated;


public:
    /* Vector objects */ // TODO: make the vectors private
	CudaVector<float>* floatBuffer;
	CudaVector<float>* windowBuffer;
	CudaVector<cufftComplex>* complexBuffer;
	CudaVector<unsigned char>* charBuffer;
    /* Initialization functions */
    CudaAlgorithm(CudaBase* obj_base, int width = 1, int height = 1, int depth = 1, int c_depth = 3);
    ~CudaAlgorithm();
    void freeMemory();
    bool initDeviceEnv();
    /* Algorithms */
    void realtimeRangeMap(float* idata, uint8_t* odata, int nof_incoming_records, int type, color_t colormap = JET);
    void rangeMap(float* idata, char* odata, int type, numKind kind, color_t colormap);
    void rangeDopplerMap(float* idata, char* odata, int type, numKind kind, color_t colormap);
    void realtimeRangeDopplerMap(float* idata, unsigned char* odata, int nof_incoming_records, int type, color_t colormap = JET);

    /* SETTER */
    void setWidth(int val){x_size = val;}
    void setHeight(int val){y_size = val;}
    void setDepth(int val){z_size = val;}
    void setDevice(CudaGPU* val){device = val;}

    /* GETTER */
    CudaVector<float>* getFloatVector(){return floatBuffer;}
	CudaVector<float>* getWindowVector(){return windowBuffer;}
	CudaVector<cufftComplex>* getComplexVector(){return complexBuffer;}
	CudaVector<unsigned char>* getCharVector(){return charBuffer;}
    int getColorDepth(){return color_depth;}
    float getWindowSize(){return windowBuffer->getSize();}
    float* getFloatBuffer(){return floatBuffer->getDevPtr();}
    float* getWindowBuffer(){return windowBuffer->getDevPtr();}
    unsigned char* getCharBuffer(){return charBuffer->getDevPtr();}
	cufftComplex* getComplexBuffer(){return complexBuffer->getDevPtr();}
    int getWidth(){return x_size;}
    int getHeight(){return y_size;}
	int getDepth(){return z_size;}
    int getImageSize(){return image_size;}

    /* MAKER */
    void make_tcp_header(tcp_header* header, int records_processed, int nof_records, int ch);
    /* Templates */
    template<typename T> void freeCudaVector(CudaVector<T>* vec){if (vec != NULL){vec->freeMemory();delete(vec);}}
    template<typename T> void saveVector(CudaVector<T>* vec, const char* filename = "results/data/processed.dat")
    {
        FILE* fid;
        T* h_buffer = (T*)malloc(vec->getSize());
        int width;
        unsigned int _size = (float)sizeof(T);
        if(sizeof(T) == 1)
            width = vec->getSize()/(color_depth*y_size);
        else
            width = vec->getSize()/(y_size*_size);

        CUDA_CHECK(cudaMemcpy(h_buffer, vec->getDevPtr(), vec->getSize(), cudaMemcpyDeviceToHost));

        fid = fopen(filename, "wb");
        fwrite((void*)&z_size, sizeof(z_size), 1, fid);
    	fwrite((void*)&y_size, sizeof(y_size), 1, fid);
        fwrite((void*)&width, sizeof(width), 1, fid);
        fwrite((void*)&_size, sizeof(_size), 1, fid);
    	fwrite((void*)h_buffer, _size, vec->getSize()/_size, fid);
        fclose(fid);
    }


};


#endif
