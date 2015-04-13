cmake_minimum_required(VERSION 2.8)
project(defect-detect)
add_definitions(-std=c++14)
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})
add_executable(defect-detect main.cpp)
find_library(PROGRAM_OPTIONS boost_program_options)
find_library(FILESYSTEM boost_filesystem)
find_library(BOOST_SYSTEM boost_system)
target_link_libraries(defect-detect ${OpenCV_LIBS} ${PROGRAM_OPTIONS} ${FILESYSTEM} ${BOOST_SYSTEM})

#Move the images to the build directory
file(GLOB IMAGES "images/*")
file(COPY ${IMAGES} DESTINATION ${CMAKE_BINARY_DIR})
