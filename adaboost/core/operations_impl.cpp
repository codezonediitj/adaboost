#ifndef ADABOOST_CORE_OPERATIONS_IMPL_CPP
#define ADABOOST_CORE_OPERATIONS_IMPL_CPP

#include<adaboost/utils/utils.hpp>
#include<adaboost/core/operations.hpp>
#include<adaboost/cuda/cuda_data_structures_impl.hpp>
#include<iostream>

namespace adaboost
{
    namespace core
    {	
	template <class data_type_vector>
        void fill(const data_type_vector value,
	          const Vector<data_type_vector>& vec)
        {
            for(unsigned int i = 0; i < vec.get_size(); i++)
            {
               vec.get_data_pointer()data[i] = value;
            }
        }

	template <class data_type_vector>
        void fill(const data_type_vector value,
                  unsigned block_size=0,
		  const VectorGPU<data_type_vector>& vec){
                if(block_size == 0)
                {
                    this->adaboost::core::Vector<data_type_vector>::fill(value);
                }
                else
                {
                    fill_vector_kernel<data_type_vector>
                    <<<
                    ( vec.get_size(gpu) + block_size - 1)/block_size,
                    block_size
                    >>>(vec.get_data_pointer(gpu), vec.get_size(gpu), value);
                }
            }

	template <class data_type_matrix>
        void fill(const data_type_matrix value,
	          const Matrix<data_type_matrix>& mat)
        {
            for(unsigned int i = 0; i < mat.get_rows(); i++)
            {
                for(unsigned int j = 0; j < mat.get_cols(); j++)
                {
                    mat.get_data_pointer()[i*mat.get_cols() + j] = value;
                }
            }
        }

	template <class data_type_matrix>
        void fill(const data_type_matrix value,
		  const MatrixGPU<data_type_matrix>& mat,
                              unsigned block_size_x=0,
                              unsigned block_size_y=0){
                if(block_size_x == 0 || block_size_y == 0)
                {
                    this->adaboost::core::Matrix<data_type_matrix>::fill(value);
                }
                else
                {
                    dim3 gridDim((mat.get_cols(gpu) + block_size_x - 1)/block_size_x,
                                  (mat.get_rows(gpu) + block_size_y - 1)/block_size_y);
                    dim3 blockDim(block_size_x, block_size_y);
                    fill_matrix_kernel<data_type_matrix>
                    <<<gridDim, blockDim>>>
                    (mat.get_data_pointer(gpu),
                     mat.get_rows(gpu),
                     value);
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
	
	template <class data_type_vector, class data_type_matrix>
        void multiply(const Vector<data_type_vector>& vec,
                      const Matrix<data_type_matrix>& mat,
                      Vector<data_type_vector>& result)
        {
            adaboost::utils::check(vec.get_size() == mat.get_rows(),
                                  "Orders mismatch in the inputs.");
            for(unsigned int j = 0; j < mat.get_cols(); j++)
            {
                data_type_vector _result = 0;
                for(unsigned int i = 0; i < mat.get_rows(); i++)
                {
                    _result += (vec.at(i)*mat.at(i, j));
                }
                result.set(j, _result);
            }
        }

        template <class data_type_matrix>
        void multiply(const Matrix<data_type_matrix>& mat1,
                      const Matrix<data_type_matrix>& mat2,
                      Matrix<data_type_matrix>& result)
        {
            adaboost::utils::check(mat1.get_cols() == mat2.get_rows(),
                                    "Order of matrices don't match.");
            unsigned int common_cols = mat1.get_cols();
            for(unsigned int i = 0; i < result.get_rows(); i++)
            {
                for(unsigned int j = 0; j < result.get_cols(); j++)
                {
                    data_type_matrix _result = 0;
                    for(unsigned int k = 0; k < common_cols; k++)
                    {
                        _result += (mat1.at(i, k)*mat2.at(k, j));
                    }
                    result.set(i, j, _result);
                }
            }
        }

        template <class data_type_vector>
                void product(const Vector<data_type_vector>& vec1,
                             const Vector<data_type_vector>& vec2,
                             data_type_vector& result)
                {
                    adaboost::utils::check(vec1.get_size() == vec2.get_size(),
                                           "Size of vectors don't match.");
                    result = 0;
                    for(unsigned int i = 0; i < vec1.get_size(); i++)
                    {
                        result += (vec1.at(i)*vec2.at(i));
                    }
                }


        #include "instantiated_templates_operations.hpp"

    } // namespace core
} // namespace adaboost

#endif
