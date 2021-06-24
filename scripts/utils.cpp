#include <iostream>
#include <stdlib.h>
#include <time.h>
#include <fstream>
#include <string>
#include <vector>
#include <numeric>
#include <algorithm>
#include <math.h>
#include "utils.hpp"
using namespace std;

// generate 2D points using pseudo-random number generator
void generate_points(int n, int seed, int range, string file_name){
    srand(seed);

    // open file handling possible errors
    ofstream points_file;
    points_file.open(file_name, ios::trunc);
    if(!points_file){
        cout<<"Error in file opening..";
        exit(-1);
    }

    // generate pseudo-random values in defined range, append in file
    for(int i = 0; i < n; i++){
        pair<float,float> point = make_pair((float)rand()/(RAND_MAX/range),(float)rand()/(RAND_MAX/range));
        points_file<<point.first<<", "<<point.second;
        if(i != n-1) points_file<<"\n";
    }
    points_file.close(); 
}

// read 2d points from file and store them into returned structure
vector<pair<float,float>> read_points(string file_name){
    ifstream points_file;
    vector<pair<float,float>> points = vector<pair<float,float>>(); // structure which will contain 2D points

    // open file handling possible errors
    points_file.open(file_name);
    if(!points_file){
        cout<<"Error in file opening..";
        exit(-1);
    }

    // each line contains a 2D point, very naive parsing..
    string line;
    while(getline(points_file,line)){
        string first = "";
        string second = "";
        bool found_comma = false;
        for(int i=0; i < line.size();i++){
            if(!found_comma){
                if(line[i]==','){
                    found_comma = true;
                    continue;
                }
                first += line[i];
            }
            else{
                second += line[i];
            }
        }
        points.push_back(make_pair(stof(first),stof(second)));
    }
    return points;
}

// write knn result to file
void write_points(string knn_results, string file_name){
    ofstream points_file;

    // open file handling possible errors
    points_file.open(file_name, ios::trunc);
    if(!points_file){
        cout<<"Error in file opening..";
        exit(-1);
    }

    // write results to file
    points_file<<knn_results;
    points_file.close();
}

// calculate distance between two 2D points
float euclidean_distance(pair<float,float> point, pair<float,float> reference){
    float euc_dist = sqrt(pow(point.first - reference.first, 2) + pow(point.second - reference.second, 2));
    return euc_dist;
}

// given a point and a set of points, calculate knn
string find_knn(vector<pair<float,float>> points, vector<pair<float,int>> best_knn_yet, int reference, int k){
    string local_result = to_string(reference) + ": "; // start building resulting string
    float dist;

    // initialize support vector to maintain top k neighbors found up to a certain point
    for(int i=0; i<k;i++){
        best_knn_yet[i].first = INT32_MAX;
        best_knn_yet[i].second = -1;
    }
    int max_nn_idx = 0; // index of neighbor in the top k with highest distance
    for(int i=0;i<points.size(); i++){
        if(i==reference) // obviously distance would be 0
            continue;
        dist = euclidean_distance(points[i], points[reference]); // get distance with current point
        if(dist < best_knn_yet[max_nn_idx].first){ // if less distant than most distant in top k --> swap and update most distant top k index
            best_knn_yet[max_nn_idx].first = dist;
            best_knn_yet[max_nn_idx].second = i;
            for(int j=0; j<k; j++){ // update most distant top k index
                if(best_knn_yet[j].first > best_knn_yet[max_nn_idx].first)
                    max_nn_idx = j;
            }
        }
        }
    sort(best_knn_yet.begin(), best_knn_yet.end()); // sort knn by dist
    for(int j=0;j<k;j++){ // finish constructing resulting string for this point
        local_result += to_string(best_knn_yet[j].second);
        local_result += ", ";
    } 
    return local_result.substr(0,local_result.length()-2); // return result avoiding final comma
}

// wrapper to calculate sequentially knn for all points in given space
string knn_seq_compute(vector<pair<float,float>> points, int k){
    string knn_results = ""; // where to construct final result
    vector<pair<float,int>> best_knn_yet(k); // save up allocation time
    for(int i=0;i<points.size();i++){
        knn_results += find_knn(points, best_knn_yet, i, k);
        knn_results += "\n";
    }
    return knn_results;
}

// check if argument given is number or not
bool isNumber(string s)
{
    for (int i = 0; i < s.length(); i++)
        if (isdigit(s[i]) == false)
            return false;
 
    return true;
}