#ifndef ADABOOST_CORE_DATA_STRUCTURES_IMPL_CPP
#define ADABOOST_CORE_DATA_STRUCTURES_IMPL_CPP

#include<adaboost/core/data_structures.hpp>
#include<adaboost/utils/utils.hpp>
#include<adaboost/memory_manager.hpp>

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
        Vector<data_type_vector>* Vector<data_type_vector>::
        create_Vector()
        {
            Vector<data_type_vector>* vec = new Vector<data_type_vector>();
            memory_manager->register_object(vec);
            return vec;
        }

        template <class data_type_vector>
        Vector<data_type_vector>* Vector<data_type_vector>::
        create_Vector(unsigned int _size)
        {
            Vector<data_type_vector>* vec = new Vector<data_type_vector>(_size);
            memory_manager->register_object(vec);
            return vec;
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
        Matrix<data_type_matrix>*  Matrix<data_type_matrix>::
        create_Matrix()
        {
            Matrix<data_type_matrix>* mat = new Matrix<data_type_matrix>();
            memory_manager->register_object(mat);
            return mat;
        }

        template <class data_type_matrix>
        Matrix<data_type_matrix>*  Matrix<data_type_matrix>::
        create_Matrix(unsigned int _rows, unsigned int _cols)
        {
            Matrix<data_type_matrix>* mat = new Matrix<data_type_matrix>(_rows, _cols);
            memory_manager->register_object(mat);
            return mat;
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

        #include "../templates/instantiated_templates_data_structures.hpp"

    } // namespace core
} // namespace adaboost

#endif
