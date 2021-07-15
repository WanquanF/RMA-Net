# acap_interp #
cmake_minimum_required (VERSION 2.8)
project(acap_interp)
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake/")
set(CMAKE_CXX_FLAGS "-std=c++11 -w -D_REENTRANT -fPIC -pipe -g -fopenmp ${CMAKE_CXX_FLAGS}")
include_directories(${CMAKE_SOURCE_DIR})
include_directories("/home/wanquan/eigen3")
include_directories("/usr/local/include")
link_directories("/usr/local/lib")


add_executable(acap_interp
    main.cpp
    util_3drotation_log_exp.cpp
    util_3drotation_log_exp.h
)

target_link_libraries(acap_interp
    -lOpenMeshCore
    -lOpenMeshTools
)

