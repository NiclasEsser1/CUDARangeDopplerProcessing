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
    int x_size;
    int y_size;
    int z_size;
    int color_depth;
    double total_required_mem;
	bool allocated;

public:
    /* Initialization functions */
    CudaAlgorithm(CudaBase* obj_base, int width = 1, int height = 1, int depth = 1, int c_depth = 3);
    ~CudaAlgorithm();
    void freeMemory();
    bool initDeviceEnv();
    /* Algorithms */
    void rangeDopplerAlgorithm(float* idata, char* odata, winType type, numKind kind, color_t colormap = JET);

    /* SETTER */
    void setWidth(int val){x_size = val;}
    void setHeight(int val){y_size = val;}
    void setDepth(int val){z_size = val;}
    void setDevice(CudaGPU* val){device = val;}

    /* GETTER */
    int getColorDepth(){return color_depth;}
    float getWindowSize(){return windowBuffer->getSize();}
    float* getFloatBuffer(){return floatBuffer->getDevPtr();}
    float* getWindowBuffer(){return windowBuffer->getDevPtr();}
    unsigned char* getCharBuffer(){return charBuffer->getDevPtr();}
	cufftComplex* getComplexBuffer(){return complexBuffer->getDevPtr();}
    int getWidth(){return x_size;}
    int getHeight(){return y_size;}
	int getDepth(){return z_size;}

    /* Vector objects */
	CudaVector<float>* floatBuffer;
	CudaVector<float>* windowBuffer;
	CudaVector<cufftComplex>* complexBuffer;
	CudaVector<unsigned char>* charBuffer;

    /* Templates */
    template<typename T> void freeCudaVector(CudaVector<T>* vec){
    	if (vec != NULL){vec->resize(0);}
    }
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
