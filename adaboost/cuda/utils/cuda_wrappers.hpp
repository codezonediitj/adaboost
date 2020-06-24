#ifndef ADABOOST_UTILS_CUDA_WRAPPERS_HPP
#define ADABOOST_UTILS_CUDA_WRAPPERS_HPP

#include<cuda.h>
#include<cuda_runtime.h>

namespace adaboost
{
    namespace utils
    {
        namespace cuda
        {
            enum direction {HostToDevice, DeviceToHost};
            typedef cudaEvent_t cuda_event_t;

            /*
            * Used for allocating memory on the device.
            *
            * @param ptr Pointer to allocated device memory
            * @param num_bytes Requested allocation size in bytes
            * allocates num_bytes of linear memory on the device
            */
            void cuda_malloc(void** ptr, unsigned num_bytes);

            /*
            * Used for copying data between host and device.
            *
            * @param ptr_1 Destination memory address
            * @param ptr_2 Source memory address
            * @param num_bytes Size in bytes to copy
            * @param d Determines the direction of copying data
            */
            void cuda_memcpy
            (void* ptr_1, void* ptr_2, unsigned num_bytes, direction d);

            /*
            * Used for creating an event object for the current device.
            *
            * @param event_ptr Newly created event
            */
            void cuda_event_create(cuda_event_t* event_ptr);

            /*
            * Used for recording an event.
            *
            * @param event Event to record
            */
            void cuda_event_record(cuda_event_t event);

            /*
            * Used for waiting for an event to complete.
            *
            * @param event Event to wait for
            */
            void cuda_event_synchronize(cuda_event_t event);

            /*
            * Used for freeing memory on the device.
            *
            * @param ptr Device pointer to memory to free
            */
            void cuda_free(void* ptr);

        } // namspace cuda
    } // namespace utils
} // namespace adaboost

#endif
