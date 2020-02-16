#ifndef ADABOOST_CORE_DATA_STRUCTURES_IMPL_CPP
#define ADABOOST_CORE_DATA_STRUCTURES_IMPL_CPP

#include<adaboost/core/data_structures.hpp>
#include<adaboost/utils/utils.hpp>

namespace adaboost
{
    namespace core
    {
        template <class data_type_vector>
        Vector<data_type_vector>::Vector():
        data(NULL),
        size(0)
        {
        }

        template <class data_type_vector>
        data_type_vector*
        Vector<data_type_vector>::_reserve_space(unsigned int _size)
        {
            adaboost::utils::check(_size > 0,
            "The size of the vector should be positive.");
            data_type_vector* pointer = new data_type_vector[_size];
            return pointer;
        }

        template <class data_type_vector>
        Vector<data_type_vector>::Vector(unsigned int _size):
        data(_reserve_space(_size)),
        size(_size)
        {
        }

        template <class data_type_vector>
        data_type_vector Vector<data_type_vector>::
        at(unsigned int index) const
        {
            adaboost::utils::check(index >= 0 && index < this->size,
                                  "Index out of range.");
            return this->data[index];
        }

        template <class data_type_vector>
        void Vector<data_type_vector>::
        set(unsigned int index, data_type_vector value)
        {
            adaboost::utils::check(index >= 0 && index < this->size,
                                  "Index out of range.");
            this->data[index] = value;
        }

        template <class data_type_vector>
        void Vector<data_type_vector>::
        fill(data_type_vector value)
        {
            for(unsigned int i = 0; i < this->size; i++)
            {
                this->data[i] = value;
            }
        }

        template <class data_type_vector>
        unsigned int Vector<data_type_vector>::get_size() const
        {
            return this->size;
        }

        template <class data_type_vector>
        data_type_vector* Vector<data_type_vector>::get_data_pointer() const
        {
            return this->data;
        }

        template <class data_type_vector>
        Vector<data_type_vector>::
        ~Vector()
        {
            if(this->data != NULL)
                delete [] this->data;
        }

        template <class data_type_matrix>
        Matrix<data_type_matrix>::
        Matrix():
        data(NULL),
        rows(0),
        cols(0)
        {
        }

        template <class data_type_matrix>
        data_type_matrix*  Matrix<data_type_matrix>::
        _reserve_space(unsigned int _rows, unsigned int _cols)
        {
            adaboost::utils::check(_rows > 0, "Number of rows should be positive.");
            adaboost::utils::check(_cols > 0, "Number of cols should be positive.");
            data_type_matrix* new_data = new data_type_matrix[_rows*_cols];
            return new_data;
        }

        template <class data_type_matrix>
        Matrix<data_type_matrix>::
        Matrix(unsigned int _rows, unsigned int _cols):
        data(_reserve_space(_rows, _cols)),
        rows(_rows),
        cols(_cols)
        {
        }

        template <class data_type_matrix>
        data_type_matrix Matrix<data_type_matrix>::
        at(unsigned int x, unsigned int y) const
        {
            adaboost::utils::check(x >= 0 && x < this->get_rows(),
                                  "Row index out of range.");
            adaboost::utils::check(y >= 0 && y < this->get_cols(),
                                 "Column index out of range.");
            return this->data[x*this->cols + y];
        }

        template <class data_type_matrix>
        void Matrix<data_type_matrix>::
        set(unsigned int x,
            unsigned int y,
            data_type_matrix value)
        {
            adaboost::utils::check(x >= 0 && x < this->get_rows(),
                                  "Row index out of range.");
            adaboost::utils::check(y >= 0 && y < this->get_cols(),
                                  "Column index out of range.");
            this->data[x*this->cols + y] = value;
        }

        template <class data_type_matrix>
        void Matrix<data_type_matrix>::
        fill(data_type_matrix value)
        {
            for(unsigned int i = 0; i < this->rows; i++)
            {
                for(unsigned int j = 0; j < this->cols; j++)
                {
                    this->data[i*this->cols + j] = value;
                }
            }
        }

        template <class data_type_matrix>
        unsigned int Matrix<data_type_matrix>::
        get_rows() const
        {
            return this->rows;
        }

        template <class data_type_matrix>
        unsigned int Matrix<data_type_matrix>::
        get_cols() const
        {
            return this->cols;
        }

        template <class data_type_matrix>
        data_type_matrix* Matrix<data_type_matrix>::get_data_pointer() const
        {
            return this->data;
        }

        template <class data_type_matrix>
        Matrix<data_type_matrix>::
        ~Matrix()
        {
            if(this->data != NULL)
            {
                delete [] this->data;
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

        #include "instantiated_templates_data_structures.hpp"

    } // namespace core
} // namespace adaboost

#endif
