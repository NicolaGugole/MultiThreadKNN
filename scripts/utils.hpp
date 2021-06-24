#ifndef UTILS_H
#define UTILS_H

#include <iostream>
#include <string>
#include <vector>

// generate 2D points using pseudo-random number generator
void generate_points(int, int, int range = 1000000, std::string file_name="points.txt");

// read 2d points from file and store them into returned structure
std::vector<std::pair<float,float>> read_points(std::string file_name="points.txt");

// write knn result to file
void write_points(std::string, std::string);

// calculate distance between two 2D points
float euclidean_distance(std::pair<float,float>, std::pair<float,float>);

// given a point and a set of points, calculate knn
std::string find_knn(std::vector<std::pair<float,float>>, std::vector<std::pair<float,int>>, int reference=0, int k = 5);

// wrapper to calculate sequentially knn for all points in given space
std::string knn_seq_compute(std::vector<std::pair<float,float>>, int);

// check if argument given is number or not
bool isNumber(std::string);


#endif

