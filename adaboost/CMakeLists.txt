include_directories(utils core)
add_library(adaboost_utils SHARED
            utils/utils_impl.cpp)
add_library(adaboost_core SHARED
            core/data_structures_impl.cpp)
if(BUILD_TESTS)
    set(EXECUTABLE_OUTPUT_PATH ${CMAKE_BINARY_DIR}/bin)
    set(LINK_LIBRARIES gtest gtest_main pthread)
    add_executable(test_data_structures tests/test_data_structures.cpp)
    target_link_libraries(test_data_structures ${LINK_LIBRARIES} adaboost_core adaboost_utils)
endif()
