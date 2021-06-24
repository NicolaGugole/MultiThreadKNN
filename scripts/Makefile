all: clean knn knn_ff

CC = g++
UTILS = utils.cpp
LIBS = -pthread -O3
LIBS_FF = -DNO_DEFAULT_MAPPING

knn: knn.cpp
	${CC} ${LIBS} ${UTILS} knn.cpp -o knn
knn_ff: knn_ff.cpp
	${CC} ${LIBS} ${LIBS_FF} ${UTILS} knn_ff.cpp -o knn_ff
clean: 
	@echo "Cleaning up..."
	@rm -rvf knn.o knn.exe knn knn_ff.o knn_ff.exe knn_ff points_*.txt 
