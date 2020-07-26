#ifndef ADABOOST_UTILS_FUNCTION_PARAMS_CU
#define ADABOOST_UTILS_FUNCTION_PARAMS_CU

namespace adaboost
{
    namespace cuda
    {
        namespace core
        {
            __device__ float cube(float x)
            {
                return -x*x*x;
            }
            
            __device__  adaboost::cuda::core::func_t<float,float> p_func_here = adaboost::cuda::core::cube;

        }
    }
}

#endif
