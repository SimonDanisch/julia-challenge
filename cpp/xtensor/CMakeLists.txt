cmake_minimum_required(VERSION 3.1)

# Project Julia Challenge
project(julia-challenge)

# Require xtensor and xsimd
find_package(xtensor 0.18.1 REQUIRED)
find_package(xsimd 7.0.0 REQUIRED)

# Force build type to Release
message(STATUS "Forcing build type to Release")
set(CMAKE_BUILD_TYPE Release CACHE STRING "Choose the type of build." FORCE)

# Compilation flags (march=native and check for c++14 compilation flag))
if (CMAKE_CXX_COMPILER_ID MATCHES "Clang" OR CMAKE_CXX_COMPILER_ID MATCHES "GNU")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=native -std=c++14")
endif()

# Sources
set(SOURCES src/main.cpp)

# Enable XSIMD acceleration
add_definitions(-DXTENSOR_USE_XSIMD)

# Setup executable
set(CHALLENGE_TARGET julia_challenge)
add_executable(${CHALLENGE_TARGET} ${SOURCES})
target_link_libraries(${CHALLENGE_TARGET} xtensor)
