#ifndef ADABOOST_CORE_DATA_STRUCTURES_HPP
#define ADABOOST_CORE_DATA_STRUCTURES_HPP

#include<adaboost/memory_manager.hpp>

namespace adaboost
{
    namespace core
    {
        /*
        * This class represents vectors used for storing data
        * in one dimension, implemented using C++ arrays.
        *
        * @tparam data_type_vector Data type of the elements
        *     supported by C++.
        */
        template <class data_type_vector>
        class Vector: public Base
        {
            private:

                //! Array for storing data internally in vectors.
                data_type_vector *data;

                //! The size of the vector.
                unsigned int size;

                /*
                * For reserving space in memory accoring to a given size.
                * Used in initializer list of parameterized constructors.
                * Returns a new pointer.
                *
                * @param _size The size for which the space is to be reserved.
                */
                static data_type_vector*
                _reserve_space(unsigned int _size);

            public:

                static Vector* create_Vector();

                static Vector* create_Vector(unsigned int _size);

                /*
                * Used for accessing the element of the Vector
                * at a given index.
                *
                * @param index The index of the element to be accessed.
                */
                data_type_vector at(unsigned int index) const;

                /*
                * Used for storing a given value at a given index.
                *
                * @param index The index at which the value is to be stored.
                * @param value The value which is to be stored.
                */
                void set(unsigned int index,
                         data_type_vector value);

                /*
                * Used for obtaining the size of the vector.
                */
                unsigned int get_size() const;

                data_type_vector* get_data_pointer() const;

                /*
                * Used for freeing memory.
                */
                virtual ~Vector();

            protected:

                /*
                * Default constructor.
                * Sets Vector::data to NULL and size to 0.
                */
                Vector();

                /*
                * Prameterized constructor.
                *
                * @param _size The size of the vector.
                *    Must be positive.
                */
                Vector(unsigned int _size);

        };

        /*
        * This class represents matrices used for storing data
        * in two dimensions, implemented using C++ arrays.
        *
        * @tparam data_type_matrix Data type of the elements
        *     supported by C++.
        */
        template <class data_type_matrix>
        class Matrix: public Base
        {
            private:

                //! Variables for storing rows and columns in the matrix.
                unsigned int rows, cols;

                //! Array for storing data internally in matrices.
                data_type_matrix* data;

                /*
                * For reserving space in memory accoring to a given rows and columns.
                * Used in initializer list of parameterized constructors.
                * Returns a new pointer.
                *
                * @param _rows Number of rows for which the space is to be reserved.
                * @param _cols Number of columns for which the space is to be reserved.
                */
                static data_type_matrix*
                _reserve_space(unsigned int _rows,
                               unsigned int _cols);

            public:

                static Matrix* create_Matrix();

                static Matrix* create_Matrix(unsigned int _rows,
                                             unsigned int _cols);

                /*
                * Used for accessing the element of the Matrix
                * at a given indices.
                *
                * @param x Row of the element to be accessed.
                * @param y Column of the element to be accessed.
                */
                data_type_matrix at(unsigned int x,
                                    unsigned int y) const;

                /*
                * Used for storing a given value at a given indices.
                *
                * @param x Row at which the value is to be stored.
                * @param y Column at which the value is to be stored.
                * @param value The value which is to be stored.
                */
                void set(unsigned int x,
                         unsigned int y,
                         data_type_matrix value);

                /*
                * Used for obtaining number of rows in the vector.
                */
                unsigned int get_rows() const;

                /*
                * Used for obtaining number of columns in the vector.
                */
                unsigned int get_cols() const;

                data_type_matrix* get_data_pointer() const;

                /*
                * Used for freeing memory.
                */
                virtual ~Matrix();

            protected:

                /*
                * Default constructor.
                * Sets Matrix::data to NULL, rows to 0
                * and cols to 0.
                */
                Matrix();

                /*
                * Prameterized constructor.
                *
                * @param _rows Number of rows in matrix.
                *    Must be positive.
                * @param _cols Number of columns in matrix.
                *    Must be positive.
                */
                Matrix(unsigned int _rows,
                       unsigned int _cols);
        };

    } // namespace core
} // namespace adaboost

#endif
