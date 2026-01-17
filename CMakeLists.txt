cmake_minimum_required(VERSION 3.16)
project(untitled C)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)

find_package(MPI REQUIRED COMPONENTS C)

add_executable(untitled
        safari.c
)

target_link_libraries(untitled PRIVATE MPI::MPI_C)
