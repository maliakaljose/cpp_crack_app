cmake_minimum_required(VERSION 2.8)
list( APPEND CMAKE_CXX_FLAGS "-std=c++0x ${CMAKE_CXX_FLAGS} -g -ggdb -ftest-coverage -fprofile-arcs -fpermissive")
project( Crack )
find_package( OpenCV REQUIRED )
find_package(Caffe REQUIRED)
file(GLOB CRACK_SRC
    "src/*.hpp"
    "src/*.cpp"
)

include_directories( ${OpenCV_INCLUDE_DIRS} ${Caffe_INCLUDE_DIRS})
add_executable( Crack ${CRACK_SRC})
target_link_libraries( Crack ${OpenCV_LIBS} ${Caffe_LIBRARIES})
