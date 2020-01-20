cmake_minimum_required(VERSION 3.10.2)
project(adaboost CXX)

# First, define all the compilation options.
option(BUILD_TESTS "Build tests." OFF)
option(INSTALL_GOOGLETEST "For installing GoogleTest on your system along with the build." ON)


# Set required standard to C++11.
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Install GoogleTest
if(INSTALL_GOOGLETEST AND BUILD_TESTS)
    include(${CMAKE_ROOT}/Modules/ExternalProject.cmake)
    ExternalProject_Add(googletest
    GIT_REPOSITORY https://github.com/google/googletest
    GIT_TAG master
    SOURCE_DIR "${CMAKE_CURRENT_BINARY_DIR}/googletest-src"
    BINARY_DIR "${CMAKE_CURRENT_BINARY_DIR}/googletest-build"
    TEST_COMMAND "")
endif()

# Set include directories
include_directories(.)

# Recurse into rest of the project
add_subdirectory(adaboost)

# Build tests
if(BUILD_TESTS)
    set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
    add_subdirectory(tests)
endif()

install(DIRECTORY adaboost DESTINATION ${CMAKE_PREFIX_PATH}
        PATTERN "CMakeLists.txt" EXCLUDE
        )
