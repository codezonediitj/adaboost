#ifndef CUDA_ADABOOST_CORE_DATA_STRUCTURES_HPP
#define CUDA_ADABOOST_CORE_DATA_STRUCTURES_HPP

#include<adaboost/core/data_structures.hpp>

namespace adaboost
{
    namespace cuda
    {
        namespace core
        {
            /*
            * This class represents the GPU version
            * of adaboost::core::Vector.
            *
            * @tparam data_type_vector Data type of the elements
            * supported by C++.
            */
            template <class data_type_vector>
            class VectorGPU: public adaboost::core::Vector<data_type_vector>
            {
                private:

                    //! Array for storing data on GPU.
                    data_type_vector* data_gpu;

                    //! The size of the vector stored on GPU.
                    unsigned size_gpu;

                    /*
                    * For reserving space in GPU memory accoring to a given size.
                    * Used in initializer list of parameterized constructors.
                    * Returns a new pointer.
                    *
                    * @param _size The size for which the space is to be reserved
                    * on GPU.
                    */
                    static data_type_vector*
                    _reserve_space_gpu(unsigned _size_gpu);

                public:

                    /*
                    * Default constructor.
                    * Sets VectorGPU::data_gpu to NULL and size_gpu to 0.
                    */
                    VectorGPU();

                    /*
                    * Prameterized constructor.
                    *
                    * @param _size The size of the vector on GPU.
                    *    Must be positive.
                    */
                    VectorGPU(unsigned _size);

                    /*
                    * Copies the data from GPU to CPU.
                    */
                    void copy_to_host();

                    /*
                    * Copies the  data from CPU to GPU.
                    */
                    void copy_to_device();

                    /*
                    * Returns the size of the vector.
                    * By default returns the size of the
                    * vector on GPU.
                    *
                    * @param gpu If true then size of the
                    *   vector on GPU otherwise size of the
                    *   vector on CPU is returned.
                    */
                    unsigned get_size(bool gpu=true) const;

                    /*
                    * Returns the data pointer, by default, of
                    * the vector on GPU.
                    *
                    * @param gpu If true then data pointer on GPU
                    *   is returned otherwise the one on CPU
                    *   is returned.
                    */
                    data_type_vector* get_data_pointer(bool gpu=true) const;

                    /*
                    * Frees the memory from both CPU and GPU.
                    */
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
