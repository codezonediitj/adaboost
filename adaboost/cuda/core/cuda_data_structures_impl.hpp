#ifndef CUDA_ADABOOST_CORE_DATA_STRUCTURES_IMPL_HPP
#define CUDA_ADABOOST_CORE_DATA_STRUCTURES_IMPL_HPP

#include<adaboost/utils/utils.hpp>
#include<adaboost/cuda/utils/cuda_wrappers.hpp>
#include<adaboost/cuda/core/cuda_data_structures.hpp>
#include<cmath>
#include<iostream>

#define MAX_BLOCK_SIZE 1024
#define BLOCK_SIZE 16

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
                adaboost::utils::cuda::cuda_malloc((void**)&new_pointer, bytes);
                return new_pointer;
            }

            template <class data_type_vector>
            VectorGPU<data_type_vector>::
            VectorGPU():
            adaboost::core::Vector<data_type_vector>(),
            size_gpu(0),
            data_gpu(NULL)
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
                adaboost::utils::cuda::cuda_memcpy(
                this->get_data_pointer(false),
                this->data_gpu,
                this->size_gpu*sizeof(data_type_vector),
                adaboost::utils::cuda::DeviceToHost);
            }

            template <class data_type_vector>
            void
            VectorGPU<data_type_vector>::copy_to_device()
            {
                adaboost::utils::cuda::cuda_memcpy(
                this->data_gpu,
                this->get_data_pointer(false),
                this->size_gpu*sizeof(data_type_vector),
                adaboost::utils::cuda::HostToDevice);
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
            VectorGPU<data_type_vector>::
            ~VectorGPU()
            {
                adaboost::utils::cuda::cuda_free(this->data_gpu);
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
                     adaboost::core::product(vec1, vec2, result);
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

            template <class data_type_matrix>
            data_type_matrix*
            MatrixGPU<data_type_matrix>::
            _reserve_space_gpu
            (unsigned _rows_gpu, unsigned _cols_gpu)
            {
                adaboost::utils::check(_rows_gpu > 0,
                "The number of rows in matrix should be positive.");
                adaboost::utils::check(_cols_gpu > 0,
                "The number of cols in matrix should be positive.");
                unsigned bytes = _rows_gpu*_cols_gpu*sizeof(data_type_matrix);
                data_type_matrix* new_pointer;
                adaboost::utils::cuda::cuda_malloc((void**)&new_pointer, bytes);
                return new_pointer;
            }

            template <class data_type_matrix>
            MatrixGPU<data_type_matrix>::
            MatrixGPU():
            adaboost::core::Matrix<data_type_matrix>(),
            rows_gpu(0),
            cols_gpu(0)
            {
            }

            template <class data_type_matrix>
            MatrixGPU<data_type_matrix>::
            MatrixGPU(unsigned _rows, unsigned _cols):
            adaboost::core::Matrix<data_type_matrix>(_rows, _cols),
            data_gpu(_reserve_space_gpu(_rows, _cols)),
            rows_gpu(_rows),
            cols_gpu(_cols)
            {
            }

            template <class data_type_matrix>
            __global__
            void fill_matrix_kernel
            (data_type_matrix* data,
             unsigned cols,
             data_type_matrix value)
            {
                unsigned row = blockDim.y*blockIdx.y + threadIdx.y;
                unsigned col = blockDim.x*blockIdx.x + threadIdx.x;
                data[row*cols + col] = value;
            }

            template <class data_type_matrix>
            void MatrixGPU<data_type_matrix>::
            fill(data_type_matrix value,
                 unsigned block_size_x,
                 unsigned block_size_y)
            {
                if(block_size_x == 0 || block_size_y == 0)
                {
                    this->adaboost::core::Matrix<data_type_matrix>::fill(value);
                }
                else
                {
                    dim3 gridDim((this->cols_gpu + block_size_x - 1)/block_size_x,
                                  (this->rows_gpu + block_size_y - 1)/block_size_y);
                    dim3 blockDim(block_size_x, block_size_y);
                    fill_matrix_kernel<data_type_matrix>
                    <<<gridDim, blockDim>>>
                    (this->data_gpu,
                     this->cols_gpu,
                     value);
                }
            }

            template <class data_type_matrix>
            void MatrixGPU<data_type_matrix>::
            copy_to_host()
            {
                adaboost::utils::cuda::cuda_memcpy(
                this->get_data_pointer(false),
                this->data_gpu,
                this->rows_gpu*this->cols_gpu*sizeof(data_type_matrix),
                adaboost::utils::cuda::DeviceToHost);
            }

            template <class data_type_matrix>
            void MatrixGPU<data_type_matrix>::
            copy_to_device()
            {
                adaboost::utils::cuda::cuda_memcpy(
                this->data_gpu,
                this->get_data_pointer(false),
                this->rows_gpu*this->cols_gpu*sizeof(data_type_matrix),
                adaboost::utils::cuda::HostToDevice);
            }

            template <class data_type_matrix>
            unsigned MatrixGPU<data_type_matrix>::
            get_rows(bool gpu) const
            {
                if(gpu)
                {
                    return this->rows_gpu;
                }
                else
                {
                    return this->adaboost::core::
                           Matrix<data_type_matrix>::get_rows();
                }
            }

            template <class data_type_matrix>
            unsigned MatrixGPU<data_type_matrix>::
            get_cols(bool gpu) const
            {
                if(gpu)
                {
                    return this->cols_gpu;
                }
                else
                {
                    return this->adaboost::core::
                           Matrix<data_type_matrix>::get_cols();
                }
            }

            template <class data_type_matrix>
            data_type_matrix* MatrixGPU<data_type_matrix>::
            get_data_pointer(bool gpu) const
            {
                if(gpu)
                {
                    return this->data_gpu;
                }
                else
                {
                    return this->adaboost::core::Matrix<data_type_matrix>::get_data_pointer();
                }
            }

            template <class data_type_matrix>
            MatrixGPU<data_type_matrix>::
            ~MatrixGPU()
            {
                adaboost::utils::cuda::cuda_free(this->data_gpu);
            }

            template <class data_type_matrix>
            __global__
            void multiply_kernel(
            data_type_matrix* mat1,
            data_type_matrix* mat2,
            data_type_matrix* result,
            unsigned mat1_rows,
            unsigned mat1_cols,
            unsigned mat2_rows,
            unsigned mat2_cols)
            {
                data_type_matrix cvalue = 0.0;
                unsigned row = blockIdx.y*blockDim.y + threadIdx.y;
                unsigned col = blockIdx.x*blockDim.x + threadIdx.x;
                if(row > mat1_rows || col > mat2_cols)
                    return ;
                for(unsigned e = 0; e < mat1_cols; e++)
                    cvalue += mat1[row*mat1_cols+e] * mat2[e*mat2_cols+col];
                result[row*mat2_cols+col] = cvalue;
            }

            template <class data_type_matrix>
            void multiply_gpu(const MatrixGPU<data_type_matrix>& mat1,
                              const MatrixGPU<data_type_matrix>& mat2,
                              MatrixGPU<data_type_matrix>& result)
            {
                adaboost::utils::check(mat1.get_cols() == mat2.get_rows(),
                                       "Order of matrices don't match.");
                dim3 gridDim((mat2.get_cols() + BLOCK_SIZE - 1)/BLOCK_SIZE,
                             (mat1.get_rows() + BLOCK_SIZE - 1)/BLOCK_SIZE);
                dim3 blockDim(BLOCK_SIZE, BLOCK_SIZE);
                multiply_kernel
                <<<gridDim, blockDim>>>
                (mat1.get_data_pointer(),
                 mat2.get_data_pointer(),
                 result.get_data_pointer(),
                 mat1.get_rows(),
                 mat1.get_cols(),
                 mat2.get_rows(),
                 mat2.get_cols());
            }

            #include "../templates/instantiated_templates_cuda_data_structures.hpp"

        } // namespace core
    } // namespace cuda
} // namespace adaboost

#endif
