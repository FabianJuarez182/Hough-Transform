all: pgm.o houghGlobal houghConst

houghGlobal: houghGlobal.cu pgm.o
    nvcc houghGlobal.cu pgm.o -o houghGlobal

houghConst: houghConst.cu pgm.o
    nvcc houghConst.cu pgm.o -o houghConst

pgm.o: common/pgm.cpp
    g++ -c common/pgm.cpp -o pgm.o
