#include "CudaAlgorithm.cuh"
#include "CudaGPU.cuh"
#include "CudaBase.cuh"
#include "CudaVector.cuh"

#include <iostream>
#include <cuda_runtime.h>
#include <stdio.h>
#include <unistd.h>



CudaAlgorithm::CudaAlgorithm(CudaBase* obj_base, int width, int height, int depth, int c_depth)
{
    base = obj_base;
    device = base->getDevice();
	x_size = width;
    y_size = height;
	z_size = depth;
    color_depth = c_depth;
	allocated = false;
	floatBuffer = NULL;
	windowBuffer = NULL;
	complexBuffer = NULL;
}

CudaAlgorithm::~CudaAlgorithm()
{
	freeMemory();
}

void CudaAlgorithm::freeMemory()
{
	// If statement just prevents misleading output
	if(allocated)
	{
		printf("Free device memory\n");
		freeCudaVector(floatBuffer);
		freeCudaVector(windowBuffer);
		freeCudaVector(complexBuffer);
		freeCudaVector(charBuffer);
        allocated = false;
	}
}


bool CudaAlgorithm::initDeviceEnv()
{
	//Allocate device memory for processing chain
	total_required_mem = (x_size * y_size * sizeof(float)*2
        + x_size * y_size * sizeof(cufftComplex)*2
        + x_size * y_size * sizeof(unsigned char)
        + x_size * sizeof(float))/(1024*1024);

    printf("\nRequired memory: %lf MBytes; total avaible memory: %ld MBytes\n", total_required_mem,device->totalMemory()/(1024*1024));
	if(total_required_mem < device->totalMemory())
	{
		floatBuffer = new CudaVector<float>(device, x_size * y_size, true);
		complexBuffer = new CudaVector<cufftComplex>(device, x_size * y_size, true);
		charBuffer = new CudaVector<unsigned char>(device, x_size * y_size * color_depth, true);
		windowBuffer = new CudaVector<float>(device, x_size, true);
		allocated = true;
		return 1;
	}
	else
	{
        floatBuffer = NULL;
        complexBuffer = NULL;
        charBuffer = NULL;
        windowBuffer = NULL;
		printf("Not enough memory avaible on the used device, aborting... \n");
		return 0;
	}

}

void CudaAlgorithm::rangeDopplerAlgorithm(float* idata, char* odata, winType type, numKind kind)
{
    CUDA_CHECK(cudaMemcpy(floatBuffer->getDevPtr(), idata, floatBuffer->getSize(), cudaMemcpyHostToDevice));
    complexBuffer->resize((x_size/2+1)*y_size);
    charBuffer->resize((x_size/2+1)*y_size*color_depth, true);

    if(kind == COMPLEX)
    {
        // base->setWindow(windowBuffer->getDevPtr(), x_size/2+1, type, kind);
        // base->hilbertTransform(floatBuffer->getDevPtr(), complexBuffer->getDevPtr(), x_size, y_size);
        // base->windowCplx(complexBuffer->getDevPtr(), windowBuffer->getDevPtr(), x_size/2+1, y_size);
        // base->c2c1dFFT(complexBuffer->getDevPtr(), x_size/2+1, y_size);
    }
    else
    {
        base->setWindow(windowBuffer->getDevPtr(), x_size, type, kind);
        base->windowReal(floatBuffer->getDevPtr(), windowBuffer->getDevPtr(), x_size, y_size);
        // base->r2c1dFFT(complexBuffer->getDevPtr(), x_size, y_size, floatBuffer->getDevPtr());
    }
    floatBuffer->resize((x_size/2+1)*y_size);
    base->transpose(complexBuffer->getDevPtr(), x_size/2+1, y_size);
    base->c2c1dFFT(complexBuffer->getDevPtr(), y_size, x_size/2+1);
    base->transpose(complexBuffer->getDevPtr(), y_size, x_size/2+1);
    base->absolute(complexBuffer->getDevPtr(), floatBuffer->getDevPtr(), x_size/2+1, y_size);
    floatBuffer->print();

    base->renderImage(floatBuffer->getDevPtr(), charBuffer->getDevPtr(), x_size/2+1, y_size, JET);
    // charBuffer->print();
    CUDA_CHECK(cudaMemcpy(odata, charBuffer->getDevPtr(), (x_size/2+1)*y_size*color_depth, cudaMemcpyDeviceToHost));
}
