#ifndef ADABOOST_UTILS_CUDA_WRAPPERS_IMPL_HPP
#define ADABOOST_UTILS_CUDA_WRAPPERS_IMPL_HPP

#include<adaboost/utils/cuda_wrappers.hpp>

namespace adaboost
{
    namespace utils
    {
        namespace cuda
        {
            void cuda_malloc(void** ptr, unsigned num_bytes)
            {
                cudaMalloc(ptr, num_bytes);
            }

            template <class data_type>
            void cuda_memcpy
            (data_type* ptr_1, data_type* ptr_2, unsigned num_bytes, direction d)
            {
                if(d == HostToDevice)
                {
                    cudaMemcpy(ptr_1, ptr_2, num_bytes, cudaMemcpyHostToDevice);
                }
                else if(d == DeviceToHost)
                {
                    cudaMemcpy(ptr_1, ptr_2, num_bytes, cudaMemcpyDeviceToHost);
                }
            }

            void cuda_event_create(cuda_event_t* event_ptr)
            {
                cudaEventCreate(event_ptr);
            }

            void cuda_event_record(cuda_event_t event)
            {
                cudaEventRecord(event);
            }

            void cuda_event_synchronize(cuda_event_t event)
            {
                cudaEventSynchronize(event);
            }

            #include "instantiated_templates_cuda_wrappers.hpp"

        } // namespace cuda
    } // namespace utils
} // namespace adaboost

#endif
