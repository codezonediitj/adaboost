#ifndef ADABOOST_UTILS_FUNCTION_PARAMS_CU
#define ADABOOST_UTILS_FUNCTION_PARAMS_CU

namespace adaboost
{
    namespace cuda
    {
        namespace core
        {
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