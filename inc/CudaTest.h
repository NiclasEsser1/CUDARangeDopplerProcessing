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
        SignalGenerator signal(500, 100, 10, x_size, y_size);
        signal.sinus();
        success = validate_getMaxValue(signal.getSignal(), x_size, y_size);
        success = validate_renderJet(signal.getSignal(), x_size, y_size);


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
    bool testCudaAlgorithms(int x_size, int y_size)
    {
        testCode success = TEST_SUCCED;
        SignalGenerator signal(1000, 100, 10, x_size, y_size);
        signal.sinus();
        signal.save();
        success = validate_rangeDoppler(signal.getSignal(), x_size, y_size);
        if(success == TEST_SUCCED)
            return true;
        else
            return false;
    }


    // SETTER AND GETTER FUNTIONS
    void setDevice(CudaGPU val){device = val;}
    CudaGPU getDevice(){return device;}
    void setTestObject(T val){object = val;}
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
        printf("\n\nTesting rangeDopplerCplxAlgorithm() function... \n");
        Bitmap_IO image(x_size/2+1, y_size);
        object->setWidth(x_size);
        object->setHeight(y_size);
        // if(object->initDeviceEnv())
        //     object->rangeDopplerAlgorithm(idata, HAMMING, COMPLEX);
        if(object->initDeviceEnv())
            object->rangeDopplerAlgorithm(idata, image.GetImagePtr(), HAMMING, REAL);
        image.Save("img/range_doppler_map.bmp");
        return TEST_SUCCED;
    }


    testCode validate_getMaxValue(float* idata, int x_size, int y_size)
    {
        printf("\n\nTesting getMaxValue() function... \n");
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
    		max = object->getMaxValue(gpu_buf->getDevPtr(), x_size, y_size);
            printf("cpu: %f; gpu: %f\n",cpu_max, max);
    	}
    	else
    	{
    		printf("GPU memory out of space, can not validate getMaxValue()...\n");
    		return TEST_OUT_OF_MEM;
    	}
        if(cpu_max != max)
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

        Bitmap_IO img(x_size, y_size);

        if(buf_size*3*sizeof(float) < device->getFreeMemory())
    	{
            // Allocate memory
            gpu_buf = new CudaVector<float>(device, buf_size);
            gpu_img = new CudaVector<unsigned char>(device, buf_size);

            // Copy test values from cpu to gpu
    		cudaMemcpy(gpu_buf->getDevPtr(), idata, gpu_buf->getSize(), cudaMemcpyHostToDevice);

            // Test function renderImage()
    	    object->renderImage(gpu_buf->getDevPtr(), gpu_img->getDevPtr(), x_size, y_size);

            // Copy processed data from device to host
            cudaMemcpy(img.GetImagePtr(), gpu_img->getDevPtr(), gpu_img->getSize(), cudaMemcpyDeviceToHost);
            img.Save("img/test.bmp");
        }
    	else
    	{
    		printf("GPU memory out of space, can not validate getMaxValue()...");
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
