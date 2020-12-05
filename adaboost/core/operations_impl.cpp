#ifndef ADABOOST_CORE_OPERATIONS_IMPL_CPP
#define ADABOOST_CORE_OPERATIONS_IMPL_CPP

#include<adaboost/utils/utils.hpp>
#include<adaboost/core/operations.hpp>
#include<cmath>

namespace adaboost
{
    namespace core
    {

        template <class data_type_vector>
        void fill(data_type_vector value, Vector<data_type_vector>* vec)
        {
            data_type_vector* vecPtr = vec->get_data_pointer();
            std::fill(vecPtr, vecPtr + vec->get_size(), value);
        }


        template <class data_type_matrix>
        void fill(data_type_matrix value, Matrix<data_type_matrix>* mat)
        {
            data_type_matrix* matPtr = mat->get_data_pointer();
            std::fill(matPtr, matPtr + mat->get_rows()*mat->get_cols(), value);
        }


        template <class data_type>
        void Sum(
        data_type (*func_ptr)(data_type),
        Vector<data_type>* vec,
        unsigned start,
        unsigned end,
        data_type& result)
        {
            adaboost::utils::check(vec->get_size() >= start,
            "Start is out of range.");
            end = vec->get_size() - 1 < end ?
                    vec->get_size() - 1 : end;
            result = 0;
            for(unsigned i = start; i <= end; i++)
            {
                result += func_ptr == NULL ? vec->at(i) : func_ptr(vec->at(i));
            }
        }

        template <class data_type_1, class data_type_2>
        void Argmax(
        data_type_2 (*func_ptr)(data_type_1),
        Vector<data_type_1>* vec,
        data_type_1& result)
        {
            data_type_2 max_val = func_ptr(vec->at(0));
            data_type_1 arg_max = vec->at(0);
            for(unsigned i = 0; i < vec->get_size(); i++)
            {
                if(max_val < func_ptr(vec->at(i)))
                {
                    max_val = func_ptr(vec->at(i));
                    arg_max = vec->at(i);
                }
            }
            result = arg_max;
        }

        template <class data_type_vector>
        void product(Vector<data_type_vector>* vec1,
                     Vector<data_type_vector>* vec2,
                     data_type_vector& result)
        {
            adaboost::utils::check(vec1->get_size() == vec2->get_size(),
                                   "Sizes of vectors don't match.");
            result = 0;
            for( unsigned i = 0; i < vec1->get_size(); i++ )
            {
                result += (vec1->at(i)*vec2->at(i));
            }
        }

        template <class data_type_vector>
        void add(Vector<data_type_vector>* vec1,
                 Vector<data_type_vector>* vec2,
                 Vector<data_type_vector>* result)
        {
            adaboost::utils::check(vec1->get_size() == vec2->get_size(),
                                   "Sizes of vectors don't match.");
            adaboost::utils::check(vec1->get_size() == result->get_size(),
                                   "Sizes of vectors don't match.");
            for( unsigned i = 0; i < vec1->get_size(); i++ )
            {
                result->set(i, vec1->at(i) + vec2->at(i));
            }
        }

        template <class data_type_vector>
        void multiply(Vector<data_type_vector>* vec,
                      data_type_vector scalar,
                      Vector<data_type_vector>* result)
        {
            adaboost::utils::check(vec->get_size() == result->get_size(),
                                   "Sizes of vectors don't match.");
            for( unsigned i = 0; i < vec->get_size(); i++ )
            {
                result->set(i, vec->at(i) * scalar);
            }
        }

        template <class data_type_vector, class data_type_matrix>
        void multiply(Vector<data_type_vector>* vec,
                     Matrix<data_type_matrix>* mat,
                     Vector<data_type_vector>* result)
        {
            adaboost::utils::check(vec->get_size() == mat->get_rows(),
                                  "Orders mismatch in the inputs.");
            for(unsigned int j = 0; j < mat->get_cols(); j++)
            {
                data_type_vector _result = 0;
                for(unsigned int i = 0; i < mat->get_rows(); i++)
                {
                    _result += (vec->at(i)*mat->at(i, j));
                }
                result->set(j, _result);
            }
        }

        template <class data_type_matrix>
        void multiply(Matrix<data_type_matrix>* mat1,
                      Matrix<data_type_matrix>* mat2,
                      Matrix<data_type_matrix>* result)
        {
            adaboost::utils::check(mat1->get_cols() == mat2->get_rows(),
                                    "Order of matrices don't match.");
            unsigned int common_cols = mat1->get_cols();
            for(unsigned int i = 0; i < result->get_rows(); i++)
            {
                for(unsigned int j = 0; j < result->get_cols(); j++)
                {
                    data_type_matrix _result = 0;
                    for(unsigned int k = 0; k < common_cols; k++)
                    {
                        _result += (mat1->at(i, k)*mat2->at(k, j));
                    }
                    result->set(i, j, _result);
                }
            }
        }

        template <class data_type_1, class data_type_2>
        void is_equal(Vector<data_type_1>* vec1,
                      Vector<data_type_1>* vec2,
                      Vector<data_type_2>* result)
        {
            for( unsigned idx = 0; idx < vec1->get_size(); idx++ )
            {
                result->set(idx, (data_type_2) (vec1->at(idx) == vec2->at(idx)));
            }
        }

        #include "../templates/instantiated_templates_operations.hpp"

    } // namespace core
} // namespace adaboost

#endif
