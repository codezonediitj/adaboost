#ifndef ADABOOST_CUDA_CORE_DATA_STRUCTURES_IMPL_HPP
#define ADABOOST_CUDA_CORE_DATA_STRUCTURES_IMPL_HPP

#include<adaboost/utils/utils.hpp>
#include<adaboost/cuda/utils/cuda_wrappers.hpp>
#include<adaboost/cuda/core/data_structures.hpp>
#include<cmath>

#define MAX_BLOCK_SIZE 1024
#define BLOCK_SIZE 16

namespace adaboost
{
    namespace cuda
    {
        namespace core
        {
            template <class data_type_vector>
            data_type_vector*
            VectorGPU<data_type_vector>::
            _reserve_space_gpu(unsigned _size_gpu)
            {
                adaboost::utils::check(_size_gpu > 0,
                "The size of the vector should be positive.");
                unsigned bytes = _size_gpu*sizeof(data_type_vector);
                data_type_vector* new_pointer;
                adaboost::utils::cuda::cuda_malloc((void**)&new_pointer, bytes);
                return new_pointer;
            }

            template <class data_type_vector>
            VectorGPU<data_type_vector>::
            VectorGPU():
            adaboost::core::Vector<data_type_vector>(),
            size_gpu(0),
            data_gpu(NULL)
            {
            }

            template <class data_type_vector>
            VectorGPU<data_type_vector>::
            VectorGPU(unsigned _size):
            adaboost::core::Vector<data_type_vector>(_size),
            data_gpu(_reserve_space_gpu(_size)),
            size_gpu(_size)
            {
            }

            template <class data_type_vector>
            VectorGPU<data_type_vector>* VectorGPU<data_type_vector>::
            create_VectorGPU()
            {
                VectorGPU<data_type_vector>* vec = new VectorGPU<data_type_vector>();
                memory_manager->register_object(vec);
                return vec;
            }

            template <class data_type_vector>
            VectorGPU<data_type_vector>* VectorGPU<data_type_vector>::
            create_VectorGPU(unsigned int _size)
            {
                VectorGPU<data_type_vector>* vec = new VectorGPU<data_type_vector>(_size);
                memory_manager->register_object(vec);
                return vec;
            }

            template <class data_type_vector>
            void
            VectorGPU<data_type_vector>::copy_to_host()
            {
                adaboost::utils::cuda::cuda_memcpy(
                this->get_data_pointer(false),
                this->data_gpu,
                this->size_gpu*sizeof(data_type_vector),
                adaboost::utils::cuda::DeviceToHost);
            }

            template <class data_type_vector>
            void
            VectorGPU<data_type_vector>::copy_to_device()
            {
                adaboost::utils::cuda::cuda_memcpy(
                this->data_gpu,
                this->get_data_pointer(false),
                this->size_gpu*sizeof(data_type_vector),
                adaboost::utils::cuda::HostToDevice);
            }

            template <class data_type_vector>
            unsigned VectorGPU<data_type_vector>::
            get_size(bool gpu) const
            {
                if(gpu)
                {
                    return this->size_gpu;
                }
                else
                {
                    return this->adaboost::core::Vector<data_type_vector>::get_size();
                }
            }

            template <class data_type_vector>
            data_type_vector* VectorGPU<data_type_vector>::
            get_data_pointer(bool gpu) const
            {
                if(gpu)
                {
                    return this->data_gpu;
                }
                else
                {
                    return this->adaboost::core::Vector<data_type_vector>::get_data_pointer();
                }
            }

            template <class data_type_vector>
            VectorGPU<data_type_vector>::
            ~VectorGPU()
            {
                adaboost::utils::cuda::cuda_free(this->data_gpu);
            }


            template <class data_type_matrix>
            data_type_matrix*
            MatrixGPU<data_type_matrix>::
            _reserve_space_gpu
            (unsigned _rows_gpu, unsigned _cols_gpu)
            {
                adaboost::utils::check(_rows_gpu > 0,
                "The number of rows in matrix should be positive.");
                adaboost::utils::check(_cols_gpu > 0,
                "The number of cols in matrix should be positive.");
                unsigned bytes = _rows_gpu*_cols_gpu*sizeof(data_type_matrix);
                data_type_matrix* new_pointer;
                adaboost::utils::cuda::cuda_malloc((void**)&new_pointer, bytes);
                return new_pointer;
            }

            template <class data_type_matrix>
            MatrixGPU<data_type_matrix>::
            MatrixGPU():
            adaboost::core::Matrix<data_type_matrix>(),
            rows_gpu(0),
            cols_gpu(0)
            {
            }

            template <class data_type_matrix>
            MatrixGPU<data_type_matrix>::
            MatrixGPU(unsigned _rows, unsigned _cols):
            adaboost::core::Matrix<data_type_matrix>(_rows, _cols),
            data_gpu(_reserve_space_gpu(_rows, _cols)),
            rows_gpu(_rows),
            cols_gpu(_cols)
            {
            }

            template <class data_type_matrix>
            MatrixGPU<data_type_matrix>*  MatrixGPU<data_type_matrix>::
            create_MatrixGPU()
            {
                MatrixGPU<data_type_matrix>* mat = new MatrixGPU<data_type_matrix>();
                memory_manager->register_object(mat);
                return mat;
            }

            template <class data_type_matrix>
            MatrixGPU<data_type_matrix>*  MatrixGPU<data_type_matrix>::
            create_MatrixGPU(unsigned int _rows, unsigned int _cols)
            {
                MatrixGPU<data_type_matrix>* mat = new MatrixGPU<data_type_matrix>(_rows, _cols);
                memory_manager->register_object(mat);
                return mat;
            }

            template <class data_type_matrix>
            void MatrixGPU<data_type_matrix>::
            copy_to_host()
            {
                adaboost::utils::cuda::cuda_memcpy(
                this->get_data_pointer(false),
                this->data_gpu,
                this->rows_gpu*this->cols_gpu*sizeof(data_type_matrix),
                adaboost::utils::cuda::DeviceToHost);
            }

            template <class data_type_matrix>
            void MatrixGPU<data_type_matrix>::
            copy_to_device()
            {
                adaboost::utils::cuda::cuda_memcpy(
                this->data_gpu,
                this->get_data_pointer(false),
                this->rows_gpu*this->cols_gpu*sizeof(data_type_matrix),
                adaboost::utils::cuda::HostToDevice);
            }

            template <class data_type_matrix>
            unsigned MatrixGPU<data_type_matrix>::
            get_rows(bool gpu) const
            {
                if(gpu)
                {
                    return this->rows_gpu;
                }
                else
                {
                    return this->adaboost::core::
                           Matrix<data_type_matrix>::get_rows();
                }
            }

            template <class data_type_matrix>
            unsigned MatrixGPU<data_type_matrix>::
            get_cols(bool gpu) const
            {
                if(gpu)
                {
                    return this->cols_gpu;
                }
                else
                {
                    return this->adaboost::core::
                           Matrix<data_type_matrix>::get_cols();
                }
            }

            template <class data_type_matrix>
            data_type_matrix* MatrixGPU<data_type_matrix>::
            get_data_pointer(bool gpu) const
            {
                if(gpu)
                {
                    return this->data_gpu;
                }
                else
                {
                    return this->adaboost::core::Matrix<data_type_matrix>::get_data_pointer();
                }
            }

            template <class data_type_matrix>
            MatrixGPU<data_type_matrix>::
            ~MatrixGPU()
            {
                adaboost::utils::cuda::cuda_free(this->data_gpu);
            }

            #include "../templates/instantiated_templates_data_structures.hpp"

        } // namespace core
    } // namespace cuda
} // namespace adaboost

#endif
