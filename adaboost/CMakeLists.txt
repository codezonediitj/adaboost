include_directories(utils core)
add_library(adaboost_utils SHARED
            utils/utils_impl.cpp)
add_library(adaboost_core SHARED
            core/data_structures_impl.cpp
            core/operations_impl.cpp
            utils/utils_impl.cpp)
if(BUILD_CUDA)
    add_library(adaboost_cuda_wrappers SHARED
                cuda/utils/cuda_wrappers_impl.cu)
    add_library(adaboost_cuda SHARED
                cuda/core/cuda_data_structures_impl.cu
                cuda/utils/cuda_wrappers_impl.cu)
endif()
if(BUILD_TESTS)
    set(EXECUTABLE_OUTPUT_PATH ${CMAKE_BINARY_DIR}/bin)
    set(LINK_LIBRARIES gtest gtest_main pthread adaboost_core)
    add_executable(test_core tests/test_core.cpp)
    target_link_libraries(test_core ${LINK_LIBRARIES})
    if(BUILD_CUDA)
        add_executable(test_cuda tests/test_cuda.cpp)
        target_link_libraries(test_cuda ${LINK_LIBRARIES} adaboost_cuda)
    endif()
endif()
