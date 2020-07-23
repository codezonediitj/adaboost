#ifndef CUDA_ADABOOST_CORE_OPERATIONS_HPP
#define CUDA_ADABOOST_CORE_OPERATIONS_HPP

#include<adaboost/cuda/core/cuda_data_structures.hpp>
using namespace adaboost::cuda::core;

namespace adaboost
{
    namespace cuda
    {
        namespace core
        {

            /*
            * This function fills the vector with a given value.
            *
            * If block size = 0, values are filled on CPU otherwise they are filled on GPU.
            *
            * @param value The value with which the vector is to be populated.
            * @param vec The Vector.
            * @param block_size Number of threads to be launched per block on GPU.
            */

            template <class data_type_vector>
            void fill(const data_type_vector value, const VectorGPU<data_type_vector>&vec, unsigned block_size);

            /*
            * This function fills the matrix with a given value.
            *
            * @param value The value with which the matrix is to be populated.
            * @param mat The Matrix.
            * @num_streams Number of streams being used to fill the matrix
            */

            template <class data_type_matrix>
            void fill(const data_type_matrix value, const MatrixGPU<data_type_matrix>& mat, unsigned num_streams);

            /*
            * This function computes
            * dot product of two vectors on
            * GPU.
            * 
            * @param vec1 First vector whose product is to be calculated
            * @param vec2 Second vector whose product is to be calculated
            * @param result Location to store answer
            * @param block_size  Number of threads to be launched per block on GPU.
            */

            template <class data_type_vector>
            void product_gpu(const VectorGPU<data_type_vector>& vec1,
            const VectorGPU<data_type_vector>& vec2,
            data_type_vector& result,
            unsigned block_size=0);

            /*
            * This function computes
            * dot product of two matrices on
            * GPU.
            * 
            * @param mat1 First matrix whose product is to be calculated
            * @param vec2 Second matrix whose product is to be calculated
            * @param result Location to store answer
            */
            template <class data_type_matrix>
            void multiply_gpu(const MatrixGPU<data_type_matrix>& mat1,
            const MatrixGPU<data_type_matrix>& mat2,
            MatrixGPU<data_type_matrix>& result);


        }// namespace core
    } // namespace cuda
} // namespace adaboost

#endif
