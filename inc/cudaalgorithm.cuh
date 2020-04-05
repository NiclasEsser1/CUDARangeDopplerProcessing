#ifndef CUDAALGORITHM_H_
#define CUDAALGORITHM_H_

#include <iostream>
#include <cuda_runtime.h>
#include <stdio.h>
#include <unistd.h>
#include <time.h>

#include "cudagpu.cuh"
#include "cudabase.cuh"
#include "cudavector.cuh"
#include "bitmap_io.h"
#include "utils.h"

using namespace std;
using namespace utils;

typedef struct{
    int alg;
    bool dop_zero_sup;
    int dop_zero_range;
    int window;
    bool window_2d;
    int color_mapping;
    int image_format;
    bool log_mapping;
    int x;
    int y;
    int z;
}processing_conf;

class CudaAlgorithm
{

private:
    CudaGPU* device;
    CudaBase* base;
    int x_size;
    int y_size;
    int z_size;
    int image_size; // Number of bytes of image
    int color_depth = 3;
    double total_required_mem;
	bool allocated;
    processing_conf *conf;
    string save_path;
public:
    /* Vector objects */ // TODO: make the vectors private
	CudaVector<float>* floatBuffer;
	CudaVector<float>* windowBuffer;
	CudaVector<cufftComplex>* complexBuffer;
	CudaVector<unsigned char>* charBuffer;
    CudaVector<short>* shortBuffer;
    /* Initialization functions */
    CudaAlgorithm(CudaGPU* dev, processing_conf *prc = nullptr);
    ~CudaAlgorithm();

    // virtual function implementation
    bool initProcessingEnv();
    void freeMemory();
    void createImage(float* idata, unsigned char* odata){}
    void insertProcessingConf();
    void printConfiguration();

    /* Algorithms */
    template <typename T> void process(T* idata, char* odata);
    void realtimeRangeMap(float* idata, char* odata, int nof_incoming_records, int window = HAMMING, int colormap = JET);
    template <typename T>void rangeMap(T* idata, char* odata);
    template <typename T>void rangeDopplerMap(T* idata, char* odata);
    void realtimeRangeDopplerMap(float* idata, char* odata, int nof_incoming_records, int window = HAMMING, int colormap = JET);

    /* SETTER */
    void setWidth(int val){x_size = val; conf->x = val;}
    void setHeight(int val){y_size = val;conf->y = val;}
    void setDepth(int val){z_size = val;conf->z = val;}
    void setDevice(CudaGPU* val){device = val;}
    void setSavePath(string s){save_path = s;}

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
    CudaBase *getCudaBase(){return base;}

    /* MAKER */
    //void make_tcp_header(tcp_header* header, int records_processed, int nof_records, int ch);
    /* Templates */
    template<typename T> void freeCudaVector(CudaVector<T>* vec){if (vec != NULL){vec->freeMemory();delete(vec);}}
    template<typename T> void save(CudaVector<T>* vec, string filename = "results/data/processed.dat");


};


#endif
