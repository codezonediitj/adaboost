#ifndef CUDA_ADABOOST_CORE_DATA_STRUCTURES_IMPL_HPP
#define CUDA_ADABOOST_CORE_DATA_STRUCTURES_IMPL_HPP

#include<adaboost/utils/utils.hpp>
#include<adaboost/cuda/cuda_data_structures.hpp>
#include<cmath>

#define MAX_BLOCK_SIZE 1024

namespace adaboost
{
    namespace cuda
    {
        namespace core
        {
            template <class data_type_vector>
            data_type_vector*
            VectorGPU<data_type_vector>::
            _reserve_space_gpu(unsigned _size_gpu)
            {
                adaboost::utils::check(_size_gpu > 0,
                "The size of the vector should be positive.");
                unsigned bytes = _size_gpu*sizeof(data_type_vector);
                data_type_vector* new_pointer;
                cudaMalloc((void**)&new_pointer, bytes);
                return new_pointer;
            }

            template <class data_type_vector>
            VectorGPU<data_type_vector>::
            VectorGPU():
            adaboost::core::Vector<data_type_vector>()
            {
            }

            template <class data_type_vector>
            VectorGPU<data_type_vector>::
            VectorGPU(unsigned _size):
            adaboost::core::Vector<data_type_vector>(_size),
            data_gpu(_reserve_space_gpu(_size)),
            size_gpu(_size)
            {
            }

            template <class data_type_vector>
            __global__ void fill_vector_kernel
            (data_type_vector* data,
             unsigned size,
             data_type_vector value)
            {
                unsigned index = threadIdx.x;
                unsigned stride = blockDim.x;
                for(unsigned i = index; i < size; i += stride)
                {
                    data[i] = value;
                }
            }

            template <class data_type_vector>
            void VectorGPU<data_type_vector>::
            fill(data_type_vector value,
                 unsigned block_size)
            {
                if(block_size == 0)
                {
                    this->adaboost::core::Vector<data_type_vector>::fill(value);
                }
                else
                {
                    adaboost::utils::check(block_size > 0,
                    "Number of threads in a block should be a positive.");
                    fill_vector_kernel<data_type_vector>
                    <<<
                    (this->size_gpu + block_size - 1)/block_size,
                    block_size
                    >>>(this->data_gpu, this->size_gpu, value);
                }
            }

            template <class data_type_vector>
            void
            VectorGPU<data_type_vector>::copy_to_host()
            {
                cudaMemcpy(this->get_data_pointer(false),
                           this->data_gpu,
                           this->size_gpu*sizeof(data_type_vector),
                           cudaMemcpyDeviceToHost);
            }

            template <class data_type_vector>
            void
            VectorGPU<data_type_vector>::copy_to_device()
            {
                cudaMemcpy(this->data_gpu,
                           this->get_data_pointer(false),
                           this->size_gpu*sizeof(data_type_vector),
                           cudaMemcpyHostToDevice);
            }

            template <class data_type_vector>
            unsigned VectorGPU<data_type_vector>::
            get_size(bool gpu) const
            {
                if(gpu)
                {
                    return this->size_gpu;
                }
                else
                {
                    return this->adaboost::core::Vector<data_type_vector>::get_size();
                }
            }

            template <class data_type_vector>
            data_type_vector* VectorGPU<data_type_vector>::
            get_data_pointer(bool gpu) const
            {
                if(gpu)
                {
                    return this->data_gpu;
                }
                else
                {
                    return this->adaboost::core::Vector<data_type_vector>::get_data_pointer();
                }
            }

            template <class data_type_vector>
            __global__
            void product_kernel
            (data_type_vector* v1, data_type_vector* v2, data_type_vector* v3,
            unsigned size)
            {
                __shared__ data_type_vector cache[MAX_BLOCK_SIZE];
                data_type_vector temp = 0;
                unsigned thread_i = threadIdx.x + blockDim.x*blockIdx.x;
                unsigned cache_i = threadIdx.x;
                while(thread_i < size)
                {
                    temp += v1[thread_i]*v2[thread_i];
                    thread_i = blockDim.x*gridDim.x;
                }
                cache[cache_i] = temp;
                __syncthreads();

                unsigned i = blockDim.x/2;
                while(i != 0)
                {
                    if(cache_i < i)
                    {
                        cache[cache_i] += cache[cache_i + i];
                    }
                    __syncthreads();
                    i /= 2;
                }

                if(cache_i == 0)
                    v3[blockIdx.x] = cache[0];
            }

            template <class data_type_vector>
            void product_gpu(const VectorGPU<data_type_vector>& vec1,
                             const VectorGPU<data_type_vector>& vec2,
                             data_type_vector& result,
                             unsigned block_size)
            {
                if(block_size == 0)
                {
                    return adaboost::core::product(vec1, vec2, result);
                }
                else
                {
                    adaboost::utils::check(vec1.get_size() == vec2.get_size(),
                                           "Size of vectors don't match.");
                    adaboost::utils::check(block_size > 0,
                    "Size of the block should be a positive multiple of 32.");
                    unsigned num_blocks = (vec1.get_size() + block_size - 1)/block_size;
                    VectorGPU<data_type_vector> temp_result(num_blocks);
                    product_kernel
                    <<<
                    num_blocks,
                    block_size
                    >>>(vec1.get_data_pointer(), vec2.get_data_pointer(),
                        temp_result.get_data_pointer(), vec1.get_size());
                    temp_result.copy_to_host();
                    result = 0;
                    for(unsigned i = 0; i < num_blocks; i++)
                    {
                        result += temp_result.at(i);
                    }
                }
            }

        } // namespace core
    } // namespace cuda
} // namespace adaboost

#endif
