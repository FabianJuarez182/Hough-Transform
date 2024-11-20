all: pgm.o houghGlobal

houghGlobal: houghGlobal.cu pgm.o
		nvcc houghGlobal.cu pgm.o -o houghGlobal -lopencv_core -lopencv_highgui -lopencv_imgcodecs -lopencv_imgproc -isystem /usr/include/opencv4

pgm.o: common/pgm.cpp
		g++ -c common/pgm.cpp -o pgm.o
