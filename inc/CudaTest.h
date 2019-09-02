#ifndef CUDATEST_H_
#define CUDATEST_H_

#include "CudaGPU.cuh"
#include "CudaKernels.cuh"
#include "CudaBase.cuh"
#include "CudaVector.cuh"
#include "Bitmap_IO.h"
#include "SignalGenerator.h"

#include <stdio.h>
//#include <stdlib.h>     /* exit, EXIT_FAILURE */
#include <iostream>
#include <typeinfo>

typedef enum {TEST_SUCCED, TEST_FAILED, TEST_OUT_OF_MEM} testCode;



template <class T>
class CudaTest
{
private:
    CudaGPU* device;
    T* object;
public:
    CudaTest(CudaGPU* dev, T* obj)
    {
        device = dev;
        object = obj;
        printf("\n\n___________________________________________\n");
        printf("Attempting to test unit instance from class %s ...\n", typeid(object).name());
    }
    ~CudaTest()
    {

    }

    bool testCudaBase(int x_size, int y_size)
    {
        testCode success = TEST_SUCCED;
        SignalGenerator signal(1000, 100, 10, x_size, y_size);
        signal.noisySinus();
        signal.save("noisy_sinus.dat");
        success = validate_max(signal.getSignal(), x_size, y_size);
        success = validate_min(signal.getSignal(), x_size, y_size);
        // success = validate_renderJet(signal.getSignal(), x_size, y_size);


        if(success == TEST_SUCCED)
            return true;
        else
            return false;

    }
    bool testCudaKernerls()
    {
        testCode success = TEST_SUCCED;
        if(success == TEST_SUCCED)
            return true;
        else
            return false;
    }
    bool testCudaVector()
    {
        testCode success = TEST_SUCCED;
        if(success == TEST_SUCCED)
            return true;
        else
            return false;
    }
    bool testCudaAlgorithms()
    {
        int x_size = object->getWidth();
        int y_size = object->getHeight();
        float fsample = 100000000;
        float fcenter = 6250000;
        float amplitude = 10;
        float bandwith = 12000000;
        float duration = 0.0001;
        float fdoppler = 5000;
        testCode success = TEST_SUCCED;
        SignalGenerator signal(fsample, fcenter, amplitude, x_size, y_size);
        signal.sweep(bandwith, duration, fdoppler);
        signal.save("sweep.dat",bandwith, duration);
        success = validate_rangeDoppler(signal.getSignal(), x_size, y_size);
        if(success == TEST_SUCCED)
            return true;
        else
            return false;
    }


    // SETTER AND GETTER FUNTIONS
    void setDevice(CudaGPU val){device = val;}
    void setTestObject(T val){object = val;}
    CudaGPU getDevice(){return device;}
    T getTestObject(){return object;}

    void printfCPUBuffer(float* buf, int width, int height)
    {
        for(int i = 0; i < height; i++)
        {
            for(int j = 0; j < width; j++)
            {
                printf("| %.1f |", buf[j+i*width]);
            }
            printf("\n");
        }
    }
protected:
    testCode validate_rangeDoppler(float* idata, int x_size, int y_size)
    {
        Bitmap_IO image_cplx(x_size/2+1, y_size, object->getColorDepth()*8);
        Bitmap_IO image_real(x_size/2+1, y_size, object->getColorDepth()*8);

        printf("\n\nTesting rangeDopplerMap(complex) function... \n");
        if(object->initDeviceEnv())
        {
            object->rangeDopplerMap(idata, image_cplx.GetImagePtr(),HAMMING, COMPLEX, JET);
            object->saveVector(object->charBuffer, "./results/data/processed_cplx.dat");
            image_cplx.Save("./results/img/colormap_examples/rdm_jet_complex.bmp");
            object->freeMemory();
        }
        sleep(1);
        printf("\n\nTesting rangeDopplerMap(real) function... \n");
        if(object->initDeviceEnv())
        {
            object->rangeDopplerMap(idata, image_real.GetImagePtr(), HAMMING, REAL, BLUE);
            object->saveVector(object->charBuffer, "results/data/processed_real.dat");
            image_real.Save("./results/img/colormap_examples/rdm_blue_real.bmp");
            object->freeMemory();
        }
        sleep(1);
        printf("\n\nTesting rangeDopplerMap(real) function... \n");
        if(object->initDeviceEnv())
        {
            object->rangeDopplerMap(idata, image_real.GetImagePtr(), HAMMING, REAL, VIRIDIS);
            object->saveVector(object->charBuffer, "results/data/processed_real.dat");
            image_real.Save("./results/img/colormap_examples/rdm_viridis_real.bmp");
            object->freeMemory();
        }
        sleep(1);
        printf("\n\nTesting rangeDopplerMap(real) function... \n");
        if(object->initDeviceEnv())
        {
            object->rangeDopplerMap(idata, image_real.GetImagePtr(), HAMMING, REAL, MAGMA);
            object->saveVector(object->charBuffer, "results/data/processed_real.dat");
            image_real.Save("./results/img/colormap_examples/rdm_magma_real.bmp");
            object->freeMemory();
        }
        sleep(1);
        printf("\n\nTesting rangeDopplerMap(real) function... \n");
        if(object->initDeviceEnv())
        {
            object->rangeDopplerMap(idata, image_real.GetImagePtr(), HAMMING, REAL, INFERNO);
            object->saveVector(object->charBuffer, "results/data/processed_real.dat");
            image_real.Save("./results/img/colormap_examples/rdm_inferno_real.bmp");
            object->freeMemory();
        }
        return TEST_SUCCED;
    }


    testCode validate_max(float* idata, int x_size, int y_size)
    {
        printf("\n\nTesting max() function... \n");
    	std::size_t buf_size =  x_size * y_size;
    	float cpu_max = 0;
        float max = 0;

        // Check if buffer size exceeds memory capacity of gpu
    	if(buf_size*2*sizeof(float) < device->getFreeMemory())
    	{
            // Allocate memory
            CudaVector<float>* gpu_buf = new CudaVector<float>(device, buf_size);

            // Determine max value on CPU
    		for(int i = 0; i < buf_size; i++)
    		{
    			if(idata[i] > cpu_max)
    				cpu_max = idata[i];
    		}
            // Copy test values from cpu to gpu
    		CUDA_CHECK(cudaMemcpy(gpu_buf->getDevPtr(), idata, gpu_buf->getSize(), cudaMemcpyHostToDevice));
            // Test function
    		max = object->max(gpu_buf->getDevPtr(), x_size, y_size);
            printf("cpu: %f; gpu: %f\n",cpu_max, max);
    	}
    	else
    	{
    		printf("GPU memory out of space, can not validate max()...\n");
    		return TEST_OUT_OF_MEM;
    	}
        if(cpu_max != max)
            return TEST_FAILED;

        printf("passed\n\n");
    	return TEST_SUCCED;
    }

    testCode validate_min(float* idata, int x_size, int y_size)
    {
        printf("\n\nTesting min() function... \n");
    	std::size_t buf_size =  x_size * y_size;
    	float cpu_min = 0;
        float min = 0;

        // Check if buffer size exceeds memory capacity of gpu
    	if(buf_size*2*sizeof(float) < device->getFreeMemory())
    	{
            // Allocate memory
            CudaVector<float>* gpu_buf = new CudaVector<float>(device, buf_size);

            // Determine max value on CPU
    		for(int i = 0; i < buf_size; i++)
    		{
    			if(idata[i] < cpu_min)
    				cpu_min = idata[i];
    		}
            // Copy test values from cpu to gpu
    		CUDA_CHECK(cudaMemcpy(gpu_buf->getDevPtr(), idata, gpu_buf->getSize(), cudaMemcpyHostToDevice));
            // Test function
    		min = object->min(gpu_buf->getDevPtr(), x_size, y_size);
            printf("cpu: %f; gpu: %f\n",cpu_min, min);
    	}
    	else
    	{
    		printf("GPU memory out of space, can not validate max()...\n");
    		return TEST_OUT_OF_MEM;
    	}
        if(cpu_min != min)
            return TEST_FAILED;

        printf("passed\n\n");
    	return TEST_SUCCED;
    }

    testCode validate_renderJet(float* idata, int x_size, int y_size)
    {
        printf("Testing renderJet() function... \n");
    	std::size_t buf_size =  x_size * y_size;

        CudaVector<float>* gpu_buf = NULL;
        CudaVector<unsigned char>* gpu_img = NULL;

        Bitmap_IO img(x_size, y_size, 8);

        if(buf_size*3*sizeof(float) < device->getFreeMemory())
    	{
            // Allocate memory
            gpu_buf = new CudaVector<float>(device, buf_size);
            gpu_img = new CudaVector<unsigned char>(device, buf_size);

            // Copy test values from cpu to gpu
    		cudaMemcpy(gpu_buf->getDevPtr(), idata, gpu_buf->getSize(), cudaMemcpyHostToDevice);

            // Test function renderImage()
    	    // object->mapColors(gpu_buf->getDevPtr(), gpu_img->getDevPtr(), x_size, y_size);

            // Copy processed data from device to host
            cudaMemcpy(img.GetImagePtr(), gpu_img->getDevPtr(), gpu_img->getSize(), cudaMemcpyDeviceToHost);
            img.Save("./results/img/test.bmp");
        }
    	else
    	{
    		printf("GPU memory out of space, can not validate renderImage()...");
    		return TEST_OUT_OF_MEM;
    	}

        printf("passed\n");
    	return TEST_SUCCED;
    }
};

// #define CHECK_TEST_RESULT(ans) {return __testSafeCall(ans, __FILE__, __LINE__); }
// inline bool __testSafeCall(testCode code, char* file, char* line)
// {
//     switch(code)
//     {
//         case TEST_SUCCED:
//             return true;
//             break;
//         case TEST_FAILED:
//             printf("The result is not as expeceted in file %s <%d>\n", __FILE__, __LINE__);
//             return false;
//             break;
//         case TEST_OUT_OF_MEM:
//             printf("Function call causes out of memory on GPU device error in file %s <%d>\n", __FILE__, __LINE__);
//             return false;
//             break;
//     }
// }

#endif
