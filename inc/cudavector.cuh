#ifndef CUDAVECTOR_CUH_
#define CUDAVECTOR_CUH_

#include "cudagpu.cuh"

#include <iostream>
#include <string>
#include <stdio.h>
#include <unistd.h>

using namespace std;

template <typename T>
class CudaVector
{
private:
    CudaGPU* device;
    T* m_bValues;
    size_t m_bSize;
    size_t type_size;
    size_t num_elements;
public:
    void* operator new(size_t size)
    {
        CudaVector<T>* object = NULL;
        object = (CudaVector<T> *)malloc(sizeof(CudaVector<T>));
        return object;
    }

    void operator delete(void* object)
    {
        free(object);
    }

    CudaVector(CudaGPU* gpu, size_t size = 1, bool print = false)
    {
        device = gpu;
        num_elements = size;
        m_bSize = size * sizeof(T);
        type_size = sizeof(T);
        if(device->checkMemory(size, print))
        {
            if(size != 0 && print)
            {
                utils::msg("Allocating memory: " + to_string((float)m_bSize/(1024*1024)) + "MBytes");
            }
            CUDA_CHECK(cudaMalloc(&m_bValues, m_bSize ));
            CUDA_CHECK(cudaMemset(m_bValues, 0, m_bSize));
        }
        else
        {
            utils::msg("GPU memory is out of range, could not allocate memory...");
            exit(1);
        }
    }

    ~CudaVector()
    {
        freeMemory();
    	// if(m_bValues != NULL)
		// CUDA_CHECK(cudaFree(m_bValues));
    }

    void freeMemory()
    {
        device->freedMemory(m_bSize);
        CUDA_CHECK(cudaFree(m_bValues));
    }

    void resize(size_t size = 1, bool print = false)
    {
        T* ptr = m_bValues;
        m_bSize = size * sizeof(T);
        type_size = sizeof(T);
        if(cudaFree(m_bValues)  == cudaSuccess)
        {
            m_bValues = ptr;
            if(device->checkMemory(size, print))
            {
                if(size != 0 && print)
                {
                    utils::msg("Resizing cuda vector..");
                    utils::msg("Allocating memory: " + to_string((float)m_bSize/(1024*1024)) + " MBytes");
                }
                CUDA_CHECK(cudaMalloc(&m_bValues, m_bSize ));
                CUDA_CHECK(cudaMemset(m_bValues, 0, m_bSize));
            }
            else
            {
                utils::msg("GPU memory is out of range, could not allocate memory..");
                exit(1);
            }
        }
    }

    void print(unsigned int first = 0)
    {
        T* cpu_buf = (T*)malloc(m_bSize);
        CUDA_CHECK(cudaMemcpy(cpu_buf, m_bValues, m_bSize, cudaMemcpyDeviceToHost));
        for(int i = first; i < num_elements; i++)
        {
            utils::msg("Buf[" + to_string(i) + "] = " + to_string(cpu_buf[i]));
        }
    }

    void save(const string filename = "vector.dat", unsigned width = 1, unsigned height = 1, unsigned depth = 1)
    {
        string dir = "./results/data/vectors/%s" + filename;
        FILE* fid = fopen(dir.c_str(), "wb");
        T* cpu_buf = (T*)malloc(m_bSize);

        CUDA_CHECK(cudaMemcpy(cpu_buf, m_bValues, m_bSize, cudaMemcpyDeviceToHost));

        if(fid != NULL)
    	{
            fwrite(&depth, sizeof(unsigned), 1, fid);
            fwrite(&height, sizeof(unsigned), 1, fid);
            fwrite(&width, sizeof(unsigned), 1, fid);
            fwrite(&type_size, sizeof(unsigned), 1, fid);
            fwrite(cpu_buf, sizeof(T), m_bSize, fid);
            fclose(fid);
        }
        else
        {
            utils::msg("Could not open file: " + dir);
        }
    }

    T* getDevPtr(int position = 0){return &m_bValues[position];}
    size_t getSize(){return m_bSize;}
    size_t gettype_size(){return type_size;}
};

#endif
