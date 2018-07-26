cmake_minimum_required(VERSION 2.8)
list( APPEND CMAKE_CXX_FLAGS "-std=c++0x ${CMAKE_CXX_FLAGS} -g -ggdb -ftest-coverage -fprofile-arcs -fpermissive -DCPU_ONLY")
project( Crack )
find_package( OpenCV REQUIRED )
find_package(Caffe REQUIRED)
file(GLOB CRACK_SRC
    "../src/*.hpp"
    "../src/*.cpp"
)
add_definitions(-D_GLIBCXX_USE_CXX11_ABI=0)
include_directories( ${OpenCV_INCLUDE_DIRS} ${Caffe_INCLUDE_DIRS})
add_executable( Crack ${CRACK_SRC})
target_link_libraries( Crack  /glob/intel-python/python3/bin/../lib/libcaffe.so.1.1.0 ${OpenCV_LIBS} ${Caffe_LIBRARIES} -lglog /glob/intel-python/versions/2018u2/intelpython3/lib/libboost_system.so.1.64.0)
