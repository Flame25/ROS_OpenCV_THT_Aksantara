cmake_minimum_required(VERSION 3.0)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON) 

project(THT_Soal8)

find_package(OpenCV REQUIRED)

add_executable(THT_Soal8 main.cpp)

target_include_directories(THT_Soal8 PRIVATE ${OpenCV_INCLUDE_DIRS})

target_link_libraries(THT_Soal8 PRIVATE ${OpenCV_LIBRARIES})
target_sources(THT_Soal8 PRIVATE main.cpp)

set(CONDA_BUILD_SYSROOT "$ENV{CONDA_PREFIX}/$ENV{HOST}/sysroot")

