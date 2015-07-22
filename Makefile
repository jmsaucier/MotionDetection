compile:
	nvcc motion.cu -arch=sm_20 -lopencv_core -lopencv_imgproc -lopencv_highgui -lopencv_objdetect

all: compile

run: compile
	./a.out
