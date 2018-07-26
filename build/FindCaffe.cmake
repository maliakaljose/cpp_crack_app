# Caffe package for CNN Triplet training
unset(Caffe_FOUND)

find_path(Caffe_INCLUDE_DIR NAMES caffe/caffe.hpp caffe/common.hpp caffe/net.hpp caffe/proto/caffe.pb.h caffe/util/io.hpp
  HINTS
  /glob/intel-python/versions/2018u2/intelpython3/pkgs/caffe-1.1.0-py36_intel_0/include)

find_library(Caffe_LIBS NAMES caffe
  HINTS
  /glob/intel-python/versions/2018u2/intelpython3/pkgs/caffe-1.1.0-py36_intel_0/lib)

if(Caffe_LIBS AND Caffe_INCLUDE_DIR)
    set(Caffe_FOUND 1)
endif()
