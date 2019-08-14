#ifndef CUDAALGORITHM_H_
#define CUDAALGORITHM_H_

#include "CudaGPU.cuh"
#include "CudaBase.cuh"
#include "CudaVector.cuh"

class CudaAlgorithm
{

private:
    CudaGPU* device;
    CudaBase* base;
	CudaVector<float>* floatBuffer;
	CudaVector<float>* windowBuffer;
	CudaVector<cufftComplex>* complexBuffer;
	CudaVector<unsigned char>* charBuffer;
    int x_size;
    int y_size;
    int z_size;
    std::size_t total_required_mem;
	bool allocated;

public:
    /* Initialization functions */
    CudaAlgorithm(CudaBase* obj_base, int width = 1, int height = 1, int depth = 1);
    ~CudaAlgorithm();
    void freeMemory();
    template<typename T> void freeCudaVector(CudaVector<T>* vec){
    	if (vec != NULL){delete(vec);}
    }
    bool initDeviceEnv();

    /* Algorithms */
    void rangeDopplerAlgorithm(float* idata, char* odata, winType type, numKind kind);

    /* SETTER */
    void setWidth(int val){x_size = val;}
    void setHeight(int val){y_size = val;}
    void setDepth(int val){z_size = val;}
    void setDevice(CudaGPU* val){device = val;}

    /* GETTER */
    float getWindowSize(){return windowBuffer->getSize();}
    float* getFloatBuffer(){return floatBuffer->getDevPtr();}
    float* getWindowBuffer(){return windowBuffer->getDevPtr();}
    unsigned char* getCharBuffer(){return charBuffer->getDevPtr();}
	cufftComplex* getComplexBuffer(){return complexBuffer->getDevPtr();}
    int getWidth(){return x_size;}
    int getHeight(){return y_size;}
	int getDepth(){return z_size;}


};


#endif
