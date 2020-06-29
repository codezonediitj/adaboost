#ifndef ADABOOST_CORE_OPERATIONS_IMPL_CPP
#define ADABOOST_CORE_OPERATIONS_IMPL_CPP

#include<adaboost/utils/utils.hpp>
#include<adaboost/core/operations.hpp>
#include<adaboost/cuda/core/cuda_data_structures_impl.hpp>
#include<iostream>

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
        void fill(const data_type_vector value, const adaboost::cuda::core::VectorGPU<data_type_vector>& vec, unsigned block_size = 0)
        {
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
        void fill(const data_type_matrix value, const adaboost::cuda::core::MatrixGPU<data_type_matrix>& mat, unsigned block_size_x = 0, unsigned block_size_y = 0)
        {
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

        #include "../templates/instantiated_templates_operations.hpp"

    } // namespace core
} // namespace adaboost

#endif
