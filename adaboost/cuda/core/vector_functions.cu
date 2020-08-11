#ifndef ADABOOST_CUDA_CORE_VECTOR_FUNCTIONS_CU
#define ADABOOST_CUDA_CORE_VECTOR_FUNCTIONS_CU

namespace adaboost
{
    namespace cuda
    {
        namespace core
        {
            /*
            * Option 1.
            */
            template <class data_type_vec, class data_type_ret>
            __device__ data_type_ret identity(data_type_vec x)
            {
                return x;
            }

            template <class data_type_vec, class data_type_ret>
            __device__
            adaboost::cuda::core::func_t<data_type_ret, data_type_vec> p_1 = identity<data_type_ret, data_type_vec>;
        }
    }
}

#endif
