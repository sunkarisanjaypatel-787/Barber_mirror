cmake_minimum_required(VERSION 3.22.1)

project("golden_vector")

add_library(
    golden_vector
    SHARED
    golden_vector.cpp
)

find_library(
    log-lib
    log
)

target_link_libraries(
    golden_vector
    ${log-lib}
)
