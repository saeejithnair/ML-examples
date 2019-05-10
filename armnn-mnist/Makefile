ARMNN_LIB = /opt/fsl-imx-xwayland/4.14-sumo-armnn-ocv/sysroots/aarch64-poky-linux/usr/lib
ARMNN_INC = /opt/fsl-imx-xwayland/4.14-sumo-armnn-ocv/sysroots/aarch64-poky-linux/usr/include

all: mnist_caffe mnist_tf

mnist_caffe: mnist_caffe.cpp mnist_loader.hpp
	${CXX} -O3 -std=c++14 -I$(ARMNN_INC) mnist_caffe.cpp -o mnist_caffe -L$(ARMNN_LIB) -larmnn -larmnnCaffeParser

mnist_tf: mnist_tf.cpp mnist_loader.hpp
	${CXX} -O3 -std=c++14 -I$(ARMNN_INC) mnist_tf.cpp -o mnist_tf -L$(ARMNN_LIB) -larmnn -larmnnTfParser -lpthread

InferenceTestImage.o: InferenceTestImage.hpp
	${CXX} -O3 -std=c++14 InferenceTestImage.cpp -o InferenceTestImage -larmnn -larmnnTfParser -lpthread

VerificationHelpers.o: VerificationHelpers.hpp
	${CXX} -O3 -std=c++14 VerificationHelpers.cpp -o VerificationHelpers -larmnn -larmnnTfParser -lpthread -shared

arc_profiler: armnn_profiler.cpp armnn_cv.hpp mnist_loader.hpp InferenceTestImage.hpp InferenceTestImage.cpp VerificationHelpers.hpp VerificationHelpers.cpp
	${CXX} -O3 -std=c++14 -I$(ARMNN_INC) armnn_profiler.cpp -o armnn_profiler -L$(ARMNN_LIB) -larmnn -larmnnTfParser -lpthread

clean:
	-rm -f mnist_tf mnist_caffe armnn_profiler

test: mnist_caffe mnist_tf
	LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:$(ARMNN_LIB) ./mnist_caffe
	LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:$(ARMNN_LIB) ./mnist_tf

