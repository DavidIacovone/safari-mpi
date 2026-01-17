cmake_minimum_required(VERSION 3.16)
project(safari C)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)

find_package(MPI REQUIRED COMPONENTS C)

add_executable(safari safari_mpi.c)
target_link_libraries(safari PRIVATE MPI::MPI_C)
