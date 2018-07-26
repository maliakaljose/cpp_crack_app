CAFFE_ROOT ?= /glob/deep-learning/versions/py-faster-rcnn/caffe-fast-rcnn

LDFLAGS +=  -L$(CAFFE_ROOT)/.build_release/lib \
			-L/usr/local/lib \
			-L/usr/lib64 \
			-L/usr/lib\
                        -L/lib64/libboost_system.so \
                        -L/lib64/libopencv_core.so \
                        -L/lib64/libopencv_highgui.so \
                        -L/lib64/libopencv_features2d.so \
	   		-L/lib64/libopencv_imgproc.so \
                        -L/glob/intel-python/versions/2018u2/intelpython3/lib \
			$(CAFFE_ROOT)/.build_release/lib/libcaffe.so

INCLUDES := -I$(CAFFE_ROOT)/include \
         	-I$(CAFFE_ROOT)/distribute/include \
         	-I$(CAFFE_ROOT)/.build_release/src \
                -I/glob/intel-python/versions/2018u2/intelpython3/include \
			-I/usr/include/opencv2

LIBRARIES := -lcaffe \
		-lprotobuf \
		-lopencv_core \
		-lopencv_highgui \
	   	-lopencv_imgproc \
                -lboost_filesystem \
	 	-lboost_system \
		-lopencv_features2d \
		-lglog

all:

	g++ -DCPU_ONLY=1 $(INCLUDES) -Ofast -g3 -Wall -fPIC -c -fmessage-length=0 -MMD -MP -MF"builttCaffe.d" -MT"builtCaffe.d" -o main.o  "main.cpp"
	g++ -DCPU_ONLY=1 $(INCLUDES) -Ofast -g3 -Wall -fPIC -c -fmessage-length=0 -MMD -MP -MF"builttCaffe.d" -MT"builtCaffe.d" -o crack.o  "crack.cpp"
	g++ -DCPU_ONLY=1 $(INCLUDES) -Ofast -g3 -Wall -fPIC -c -fmessage-length=0 -MMD -MP -MF"builttCaffe.d" -MT"builtCaffe.d" -o predict.o  "predict.cpp"
	g++ -o "builtCaffe" main.o predict.o crack.o  $(LDFLAGS) $(LIBRARIES)

        main.o: main.cpp 
		g++ -c main.cpp
        crack.o: crack.cpp crack.hpp
		g++ -c crack.cpp crack.hpp
        predict.o: predict.cpp predict.hpp
		g++ -c predict.cpp predict.hpp

