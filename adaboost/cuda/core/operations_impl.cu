#ifndef CUDA_ADABOOST_CORE_OPERATIONS_IMPL_HPP
#define CUDA_ADABOOST_CORE_OPERATIONS_IMPL_HPP

#include<adaboost/utils/utils.hpp>
#include<adaboost/cuda/core/operations.hpp>
#include<adaboost/cuda/utils/cuda_wrappers.hpp>
#include<adaboost/cuda/core/cuda_data_structures_impl.hpp>
#include<adaboost/core/operations_impl.cpp>
#include<iostream>
#include<cmath>

namespace adaboost
{
    namespace cuda
    {
        namespace core
        {
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
            void fill(data_type_vector value, VectorGPU<data_type_vector>& vec, unsigned block_size)
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
            __global__ void fill_matrix_kernel
            (data_type_matrix* data, unsigned cols, data_type_matrix value)
            {
                unsigned row = blockDim.y*blockIdx.y + threadIdx.y;
                unsigned col = blockDim.x*blockIdx.x + threadIdx.x;
                data[row*cols + col] = value;
            }

            template <class data_type_matrix>
            void fill(data_type_matrix value, MatrixGPU<data_type_matrix>& mat, unsigned block_size_x, unsigned block_size_y)
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

            template <class data_type_matrix>
            __global__
            void thread_multi(data_type_matrix *t1, data_type_matrix value, unsigned total_elements)
            {
                unsigned i = blockDim.x * blockIdx.x + threadIdx.x;
                if(i < total_elements)
                {
                    t1[i] = value;
                }
            }

            template <class data_type_matrix>
            void fill(data_type_matrix value, MatrixGPU<data_type_matrix>& mat, unsigned num_streams)
            {
                adaboost::utils::check(num_streams != 0, "Need positive number of streams");

                if(num_streams > mat.get_rows())
                {
                    num_streams = mat.get_rows();
                }

                bool gpu = true;
                unsigned N = mat.get_cols(), total_elements = mat.get_cols()*mat.get_rows();

                adaboost::utils::cuda::cuda_stream_t stream[num_streams];

                for(int i = 0; i < num_streams; i++)
                {
                    adaboost::utils::cuda::cuda_stream_create(&stream[i]);
                }
                for(int i = 0; i < num_streams; i++)
                {
                    adaboost::utils::cuda::cuda_stream_synchronize(stream[i]);
                }

                for(unsigned row = 0; row < mat.get_rows(); row++)
                {
                    int curr_stream = row % num_streams;
                    thread_multi<<<1, N, 0, stream[curr_stream]>>>(mat.get_data_pointer(gpu) + row*N, value, total_elements);
                }

                for(int i = 0;i < num_streams; i++)
                {
                    adaboost::utils::cuda::cuda_stream_synchronize(stream[i]);
                }
                for (int i = 0; i < num_streams; i++)
                {
                    adaboost::utils::cuda::cuda_stream_destroy(stream[i]);
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
            void product_gpu(VectorGPU<data_type_vector>& vec1,
                             VectorGPU<data_type_vector>& vec2,
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
            __device__
            data_type_matrix get_element(
            data_type_matrix* mat,
            unsigned row,
            unsigned col,
            unsigned stride)
            {
                return mat[row*stride+col];
            }

            template <class data_type_matrix>
            __device__
            void set_element(
            data_type_matrix* mat,
            unsigned row,
            unsigned col,
            data_type_matrix value,
            unsigned stride)
            {
                mat[row*stride+col] = value;
            }

            template <class data_type_matrix>
            __device__
            data_type_matrix* get_sub_matrix(
            data_type_matrix* mat,
            unsigned block_row,
            unsigned block_col,
            unsigned stride)
            {
                data_type_matrix* mat_sub =
                new data_type_matrix[BLOCK_SIZE*BLOCK_SIZE];
                mat_sub = &mat[stride*BLOCK_SIZE*block_row+BLOCK_SIZE*block_col];
                return mat_sub;
            }

            template <class data_type_matrix>
            __global__
            void multiply_kernel(
            data_type_matrix* mat1,
            data_type_matrix* mat2,
            data_type_matrix* result,
            unsigned mat1_cols,
            unsigned mat1_rows,
            unsigned mat2_cols,
            unsigned mat2_rows,
            unsigned result_cols,
            unsigned result_rows)
            {
                unsigned block_row = blockIdx.y;
                unsigned block_col = blockIdx.x;
                data_type_matrix* result_sub = get_sub_matrix(result, block_row,
                                                              block_col, result_cols);

                unsigned row = threadIdx.y;
                unsigned col = threadIdx.x;

                __shared__ data_type_matrix mat1_shared[BLOCK_SIZE][BLOCK_SIZE];
                __shared__ data_type_matrix mat2_shared[BLOCK_SIZE][BLOCK_SIZE];
                data_type_matrix cvalue = 0.0;

                for(unsigned m = 0; m < (mat1_cols + BLOCK_SIZE - 1)/BLOCK_SIZE; m++)
                {
                    data_type_matrix* mat1_sub = get_sub_matrix(mat1, block_row,
                                                                m, mat1_cols);
                    data_type_matrix* mat2_sub = get_sub_matrix(mat2, m,
                                                                block_col, mat2_cols);


                    if (m*BLOCK_SIZE + col < mat1_cols && (block_row*BLOCK_SIZE+ row) < mat1_rows)
                        mat1_shared[row][col] = get_element(mat1_sub, row, col, mat1_cols);
                    else
                        mat1_shared[row][col]=0;

                    if (m*BLOCK_SIZE + row < mat2_rows && (block_col*BLOCK_SIZE+col) < mat2_cols)
                        mat2_shared[row][col] = get_element(mat2_sub, row, col, mat2_cols);
                    else
                        mat2_shared[row][col]=0;

                    __syncthreads();

                    for(unsigned e = 0; e < BLOCK_SIZE; e++)
                    {
                        cvalue += mat1_shared[row][e] * mat2_shared[e][col];
                    }

                    __syncthreads();

                }
                if(block_row*BLOCK_SIZE+ row<result_rows && block_col*BLOCK_SIZE+col<result_cols)
                    set_element(result_sub, row, col, cvalue, result_cols);

            }

            template <class data_type_matrix>
            void multiply_gpu(MatrixGPU<data_type_matrix>& mat1,
                              MatrixGPU<data_type_matrix>& mat2,
                              MatrixGPU<data_type_matrix>& result)
            {
                adaboost::utils::check(mat1.get_cols() == mat2.get_rows(),
                                       "Order of matrices don't match.");
                dim3 gridDim((mat2.get_cols() + BLOCK_SIZE)/BLOCK_SIZE,
                             (mat1.get_rows() + BLOCK_SIZE)/BLOCK_SIZE);
                dim3 blockDim(BLOCK_SIZE, BLOCK_SIZE);
                multiply_kernel
                <<<gridDim, blockDim>>>
                (mat1.get_data_pointer(),
                 mat2.get_data_pointer(),
                 result.get_data_pointer(),
                 mat1.get_cols(),
                 mat1.get_rows(),
                 mat2.get_cols(),
                 mat2.get_rows(),
                 result.get_cols(),
                 result.get_rows());
            }

            #include "../templates/instantiated_templates_cuda_operations.hpp"

        } //namespace core
    } //namespace cuda
} //namespace adaboost
#endif
