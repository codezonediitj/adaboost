#ifndef CUDA_ADABOOST_CORE_DATA_STRUCTURES_HPP
#define CUDA_ADABOOST_CORE_DATA_STRUCTURES_HPP

#include<adaboost/core/data_structures.hpp>

namespace adaboost
{
    namespace cuda
    {
        namespace core
        {
            template <class data_type_vector>
            class VectorGPU: public adaboost::core::Vector<data_type_vector>
            {
                private:

                    data_type_vector* data_gpu;

                    unsigned size_gpu;

                    static data_type_vector*
                    _reserve_space_gpu(unsigned _size_gpu);

                public:

                    VectorGPU();

                    VectorGPU(unsigned _size);

                    void fill(data_type_vector value,
                              unsigned block_size=0);

                    void copy_to_host();

                    void copy_to_device();

                    unsigned get_size(bool gpu=true) const;

                    data_type_vector* get_data_pointer(bool gpu=true) const;

                    ~VectorGPU();
            };

            template <class data_type_vector>
            void product_gpu(const VectorGPU<data_type_vector>& vec1,
                             const VectorGPU<data_type_vector>& vec2,
                             data_type_vector& result,
                             unsigned block_size=0);

            template <class data_type_matrix>
            class MatrixGPU: public adaboost::core::Matrix<data_type_matrix>
            {
                private:

                    data_type_matrix* data_gpu;

                    unsigned rows_gpu, cols_gpu;

                    static data_type_matrix* _reserve_space_gpu
                    (unsigned _rows_gpu, unsigned _cols_gpu);

                public:

                    MatrixGPU();

                    MatrixGPU(unsigned _rows, unsigned _cols);

                    void fill(data_type_matrix value,
                              unsigned block_size_x=0,
                              unsigned block_size_y=0);

                    void copy_to_host();

                    void copy_to_device();

                    unsigned get_rows(bool gpu=true) const;

                    unsigned get_cols(bool gpu=true) const;

                    data_type_matrix* get_data_pointer(bool gpu=true) const;

                    ~MatrixGPU();
            };

            template <class data_type_matrix>
            void multiply_gpu(const MatrixGPU<data_type_matrix>& mat1,
                              const MatrixGPU<data_type_matrix>& mat2,
                              MatrixGPU<data_type_matrix>& result);

        } // namespace core
    } // namespace cuda
} // namespace adaboost

#endif
