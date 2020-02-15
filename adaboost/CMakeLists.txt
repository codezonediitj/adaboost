include_directories(utils core)
add_library(adaboost_utils SHARED
            utils/utils_impl.cpp)
add_library(adaboost_core SHARED
            core/data_structures_impl.cpp
            core/operations_impl.cpp)
if(BUILD_TESTS)
    set(EXECUTABLE_OUTPUT_PATH ${CMAKE_BINARY_DIR}/bin)
    set(LINK_LIBRARIES gtest gtest_main pthread adaboost_core adaboost_utils)
    add_executable(test_core tests/test_core.cpp)
    target_link_libraries(test_core ${LINK_LIBRARIES})
endif()
