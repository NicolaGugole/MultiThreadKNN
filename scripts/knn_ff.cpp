#include <iostream>
#include <fstream>
#include <ff/ff.hpp>
#include <ff/parallel_for.hpp>
#include <vector>
#include <math.h>
#include <chrono>
#include <numeric>
#include <string>
#include "utils.hpp" // containing reading, writing, computing utilities
#include "utimer.cpp"
using namespace std;


int main(int argc, char * argv[]) {
    if(argc == 1) {
        cout << "Usage is: ./knn_ff n [seed] [pardegree] [k]" << endl;
        cout << "Or alternatevely.."<<endl;
        cout << "Usage is: ./knn_ff filename [pardegree] [k]" << endl;
        return(0);
    }

    long seq_time, par_time; // for later speedup print
    
    // get number of workers and number of neighbors requested
    int nw = (!isNumber(argv[1]) && argc>2) ? atoi(argv[2]) : (argc>3 ? atoi(argv[3]) : 1);
    int k = (!isNumber(argv[1]) && argc>3) ? atoi(argv[3]) : (argc>4 ? atoi(argv[4]) : 5);
    vector<pair<float,float>> points; // where 2d points are stored
    string knn_seq_results;           // what will store the sequential result, string for convenience of output to file
    string knn_par_results = "";      // what will store the  parallel  result, string for convenience of output to file
    string filename = "points.txt";   // default filename to read from

    // file given as input, no generation
    if(!isNumber(argv[1])){
        filename = argv[1];
        cout<<"KNN read from "<<filename<<endl;
    }
    else{ // file not given as input --> generate points, read, compute, write..
        int points_count = atoi(argv[1]); // number of points to be generated
        int seed = 123; // the perfect seed
        if(argc > 2)
            seed = atoi(argv[2]);

        cout<<"KNN for "<<points_count<<" randomly generated 2D points"<<endl;
        {
            utimer timer("generate_points");
            generate_points(points_count, seed); // generate points
        }
    }

    // read points and store them accordingly
    {
        utimer timer("read_points");
        points = read_points(filename);
    }

    // check if points are more than neighbors required
    if(k>points.size()-1)
        k = points.size()-1;
    
    // sequential execution
    {                                           
        utimer timer("knn sequential computation", &seq_time);
        knn_seq_results=knn_seq_compute(points, k);
    }
 
    // FastFlow parallel execution
    {
    utimer tp("knn parallel ff computation", &par_time);
    
    // identity for reduce variable 
    string identity = "";

    // no need to instantiate object since this is a one-shot, avoid overhead of creating ParallelForReduce object
    ff::parallel_reduce(knn_par_results,identity,
                    0, points.size(),
                    1,
                    0, // static partitioning
                    [&](const long i,string &local_result){ 
                        vector<pair<float,int>> best_knn_yet(k);
                        local_result += find_knn(points, best_knn_yet, i, k); // local reduce
                        local_result += "\n";
                    },
                    [](string& s, const string& d) { s+=d;}, // global reduce
                    nw);
    // no need to sort because of static partitioning
    }

    // write results to file
    {
        utimer tp("write point");
        write_points(knn_seq_results, "points_knn_seq.txt");
        write_points(knn_par_results, "points_knn_ff_par.txt");
    }
    cout<<"\nSpeedup("<<nw<<"): "<<(float)seq_time/par_time<<endl;
    return(0);
}

