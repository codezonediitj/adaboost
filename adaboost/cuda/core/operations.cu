#ifndef ADABOOST_CUDA_CORE_OPERATIONS_CU
#define ADABOOST_CUDA_CORE_OPERATIONS_CU

#include<adaboost/cuda/core/cuda_data_structures.hpp>

namespace adaboost
{
    namespace cuda
    {
        namespace core
        {

            template <typename data_type_vec, typename data_type_ret>
            using func_t = data_type_ret(*)(data_type_vec);

            template <class data_type_vec, class data_type_ret>
            void Argmax(
            func_t<data_type_vec,data_type_ret> p_func,
            const VectorGPU<data_type_vec>& vec,
            unsigned& result,
            unsigned int block_size);

        }// namespace core
    } // namespace cuda
} // namespace adaboost

#endif
