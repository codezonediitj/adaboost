include_directories(utils core)
add_library(adaboost_utils SHARED
            utils/utils_impl.cpp)
add_library(adaboost_core SHARED
            core/data_structures_impl.cpp)
if(BUILD_CUDA)
    add_library(adaboost_cuda SHARED
                cuda/cuda_data_structures_impl.cu)
endif()
if(BUILD_TESTS)
    set(EXECUTABLE_OUTPUT_PATH ${CMAKE_BINARY_DIR}/bin)
    set(LINK_LIBRARIES gtest gtest_main pthread adaboost_core adaboost_utils)
    add_executable(test_data_structures tests/test_data_structures.cpp)
    target_link_libraries(test_data_structures ${LINK_LIBRARIES})
    if(BUILD_CUDA)
        add_executable(test_cuda_data_structures test_cuda_data_structures.cu)
        target_link_libraries(test_cuda_data_structures ${LINK_LIBRARIES} adaboost_cuda)
    endif()
endif()
