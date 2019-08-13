#ifndef CUDAVECTOR_CUH_
#define CUDAVECTOR_CUH_

#include "CudaGPU.cuh"
#include <iostream>
#include <string>
#include <cuda_runtime.h>
#include <cufft.h>
#include <stdio.h>
#include <unistd.h>



template <typename T>
class CudaVector
{
private:
    CudaGPU* device;
    T* m_bValues;
    std::size_t m_bSize;
    std::size_t eleSize;
public:
    __host__
    void* operator new(std::size_t size)
    {
        CudaVector<T>* object = NULL;
        object = (CudaVector<T> *)malloc(sizeof(CudaVector<T>));
        return object;
    }

    __host__
    void operator delete(void* object)
    {
        // if(m_bValues != NULL)
        //     CUDA_CHECK(cudaFree(m_bValues));
        free(object);
    }

    __host__
    CudaVector(CudaGPU* gpu, std::size_t size = 1, bool print = false)
    {
        device = gpu;
        m_bSize = size * sizeof(T);
        eleSize = sizeof(T);
        if(device->checkMemory(size, print))
        {
            if(size != 0 && print)
                printf("Allocating memory: %.4f MBytes\n", (float)m_bSize/(1024*1024));
            CUDA_CHECK(cudaMalloc(&m_bValues, m_bSize ));
            CUDA_CHECK(cudaMemset(m_bValues, 0, m_bSize));
        }
        else
        {
            printf("GPU memory is out of range, could not allocate memory..\n");
            exit(1);
        }
    }

    __host__
    ~CudaVector()
    {
    	// if(m_bValues != NULL)
    	// 	CUDA_CHECK(cudaFree(m_bValues));
    }

    void resize(std::size_t size = 1, bool print = false)
    {
        printf("Resizing cuda vector...\n");
        CUDA_CHECK(cudaFree(m_bValues));
        // sleep(3);
        m_bSize = size * sizeof(T);
        eleSize = sizeof(T);

        if(device->checkMemory(size, print))
        {
            if(size != 0 && print)
                printf("Allocating memory: %.4f MBytes\n", (float)m_bSize/(1024*1024));
            CUDA_CHECK(cudaMalloc(&m_bValues, m_bSize ));
            CUDA_CHECK(cudaMemset(m_bValues, 0, m_bSize));
        }
        else
        {
            printf("GPU memory is out of range, could not allocate memory..\n");
            exit(1);
        }
    }
    __host__
    T* getDevPtr(int position = 0)
    {
        return &m_bValues[position];
    }

    __host__
    std::size_t getSize()
    {
        return m_bSize;
    }

    __host__
    std::size_t geteleSize()
    {
        return eleSize;
    }

    __host__
    void printComplex()
    {
        cufftComplex* cpu_buf = (cufftComplex*)malloc(m_bSize);
        CUDA_CHECK(cudaMemcpy(cpu_buf, m_bValues, m_bSize, cudaMemcpyDeviceToHost));
        for(int i = 0; i < m_bSize/sizeof(T); i++)
        {
            std::cout << "Buf[" << i << "] = " << cpu_buf[i].x << " + i * " << cpu_buf[i].y << std::endl;
        }
    }
    __host__
    void print()
    {
        T* cpu_buf = (T*)malloc(m_bSize);
        CUDA_CHECK(cudaMemcpy(cpu_buf, m_bValues, m_bSize, cudaMemcpyDeviceToHost));
        for(int i = 0; i < m_bSize/sizeof(T); i++)
        {
            std::cout << "Buf[" << i << "] = " << cpu_buf[i] << std::endl;
        }
    }
};

#endif
