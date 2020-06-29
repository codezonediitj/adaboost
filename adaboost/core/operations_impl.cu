#ifndef ADABOOST_CORE_OPERATIONS_IMPL_CPP
#define ADABOOST_CORE_OPERATIONS_IMPL_CPP

#include<adaboost/utils/utils.hpp>
#include<adaboost/core/operations.hpp>
#include<iostream>
#include<adaboost/cuda/utils/cuda_wrappers.hpp>
#include<adaboost/cuda/core/cuda_data_structures.hpp>
#include<cmath>
using namespace adaboost::cuda::core;

namespace adaboost
{
    namespace core
    {

        template <class data_type_vector>
        void fill(const data_type_vector value, const Vector<data_type_vector>&vec)
        {
            for(unsigned i = 0; i < vec.get_size(); i++)
            {
                vec.get_data_pointer()[i] = value;
            }
        }

        template <class data_type_vector>
        __global__ void fill_vector_kernel
        (data_type_vector* data, unsigned size, data_type_vector value)
        {
            unsigned index = threadIdx.x;
            unsigned stride = blockDim.x;
            for(unsigned i = index; i < size; i += stride)
            {
                data[i] = value;
            }
        }

        template <class data_type_vector>
        void fill(const data_type_vector value, const VectorGPU<data_type_vector>& vec, unsigned block_size)
        {
            bool gpu=true;
            if(block_size == 0)
            {
                adaboost::core::fill(value, vec);
            }
            else
            {
                fill_vector_kernel<data_type_vector>
                <<<
                (vec.get_size(gpu) + block_size - 1)/block_size, block_size
                >>>
                (vec.get_data_pointer(gpu), vec.get_size(gpu), value);
            }
        }

        template <class data_type_matrix>
        void fill(data_type_matrix value, const Matrix<data_type_matrix>&mat)
        {
            for(unsigned i = 0; i < mat.get_rows(); i++)
            {
                for(unsigned j = 0; j < mat.get_cols(); j++)
                {
                    mat.get_data_pointer()[i*mat.get_cols() + j] = value;
                }
            }
        }

        template <class data_type_matrix>
        __global__ void fill_matrix_kernel
        (data_type_matrix* data, unsigned cols, data_type_matrix value)
        {
            unsigned row = blockDim.y*blockIdx.y + threadIdx.y;
            unsigned col = blockDim.x*blockIdx.x + threadIdx.x;
            data[row*cols + col] = value;
        }

        template <class data_type_matrix>
        void fill(const data_type_matrix value, const MatrixGPU<data_type_matrix>& mat, unsigned block_size_x, unsigned block_size_y)
        {
            bool gpu=true;
            if(block_size_x == 0 || block_size_y == 0)
            {
                adaboost::core::fill(value, mat);
            }
            else
            {
                dim3 gridDim((mat.get_cols(gpu) + block_size_x - 1)/block_size_x, (mat.get_rows(gpu) + block_size_y - 1)/block_size_y);
                dim3 blockDim(block_size_x, block_size_y);
                fill_matrix_kernel<data_type_matrix>
                <<<
                gridDim, blockDim
                >>>
                (mat.get_data_pointer(gpu), mat.get_rows(gpu), value);
            }
        }

        template <class data_type>
        void Sum(
        data_type (*func_ptr)(data_type),
        const Vector<data_type>& vec,
        unsigned start,
        unsigned end,
        data_type& result)
        {
            adaboost::utils::check(vec.get_size() >= start,
            "Start is out of range.");
            end = vec.get_size() - 1 < end ?
                    vec.get_size() - 1 : end;
            result = 0;
            for(unsigned i = start; i <= end; i++)
            {
                result += func_ptr(vec.at(i));
            }
        }

        template <class data_type_1, class data_type_2>
        void Argmax(
        data_type_2 (*func_ptr)(data_type_1),
        const Vector<data_type_1>& vec,
        data_type_1& result)
        {
            data_type_2 max_val = func_ptr(vec.at(0));
            data_type_1 arg_max = vec.at(0);
            for(unsigned i = 0; i < vec.get_size(); i++)
            {
                if(max_val < func_ptr(vec.at(i)))
                {
                    max_val = func_ptr(vec.at(i));
                    arg_max = vec.at(i);
                }
            }
            result = arg_max;
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
            
        #include "../templates/instantiated_templates_operations.hpp"

    } // namespace core
} // namespace adaboost

#endif
