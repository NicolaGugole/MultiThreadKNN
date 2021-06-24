#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <chrono>
#include <thread>
#include <numeric>
#include <algorithm>
#include <math.h>
#include "utimer.cpp"
#include "utils.hpp" // containing reading, writing, computing utilities

using namespace std;

int main(int argc, char *argv[]) {
    if(argc == 1) {
        cout << "Usage is: ./knn n [seed] [pardegree] [k]" << endl;
        cout << "Or alternatevely.."<<endl;
        cout << "Usage is: ./knn filename [pardegree] [k]" << endl;
        return(0);
    }

    long seq_time, par_time; // for later speedup print

    // get number of workers and number of neighbors requested
    int nw = (!isNumber(argv[1]) && argc>2) ? atoi(argv[2]) : (argc>3 ? atoi(argv[3]) : 1);
    int k = (!isNumber(argv[1]) && argc>3) ? atoi(argv[3]) : (argc>4 ? atoi(argv[4]) : 5);
    vector<pair<float,float>> points; // where 2d points are stored
    string knn_seq_results = "";      // what will store the sequential result, string for convenience of output to file
    string knn_par_results = "";      // what will store the  parallel  result, string for convenience of output to file
    vector<string> local_results(nw); // what will store the local result of each worker, indexed by thid, reducing all computations in nw strings
    string filename = "points.txt";   // default filename to read from

    // function to be assigned in parallel execution, each thread gets assigned a range of index to compute
    auto knn_batch_compute_par = [&points,&local_results](const int thid, const pair<int,int> range, const int k){ 
        string local_result = "";
        vector<pair<float,int>> best_knn_yet(k); // support vector maintaining k nearest neighbors during algorithm
        for(int i=range.first;i<range.second;i++){
            local_result += find_knn(points, best_knn_yet, i, k); // compute + local reduce
            local_result += "\n";
        }
        local_results[thid]=local_result;
        return;
    };

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
    
    // parallel execution
    {                                          
        utimer timer("knn parallel computation", &par_time);
        vector<thread> threads;

        
        // variables needed for having ranges and splitting them
        vector<pair<int,int>> ranges(nw);
        int delta = points.size() / nw;
    
        // split the work in chunks
        for(int i=0; i<nw; i++)
            ranges[i] = make_pair(i*delta,(i != (nw-1) ? (i+1)*delta : points.size()));
        

        // let threads start, assigning them a function and an amount of work
        for(int i=0; i<nw; i++)
            threads.push_back(thread(knn_batch_compute_par, i, ranges[i], k));
        
        // wait for thread to terminate
        for(thread& t: threads)                         
            t.join();

        
        // final global reduce
        for(int i=0; i<nw; i++)
            knn_par_results += local_results[i];
    }
    
    // write results to file
    {
        utimer tp("write point");
        write_points(knn_seq_results, "points_knn_seq.txt");
        write_points(knn_par_results, "points_knn_par.txt");
    }
    cout<<"\nSpeedup("<<nw<<"): "<<(float)seq_time/par_time<<endl;
    return 0;
}