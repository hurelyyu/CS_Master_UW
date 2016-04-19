//
//  measure of sortedness
//
//  Created by yyq on 15/10/23.
// measure of sortedness can be achieve by using Count Inversions algorithm
#include <fstream>
#include <iostream>
using namespace std;

const int n = 300;

int cinversion(int ary[], int n){     //for all integer in array, if there is one biger than the interger behind
    int counter=0;                    // it, this is one misorder, counter + 1
    int i, j;
    for(i = 0; i<n-1; i++)
        for(j = i+1; j<n; j++)
            if(ary[i]>ary[j])
            counter = counter + 1;
            return counter;
}

int main() {
   
    int input[n];
    
    ifstream infile;
    infile.open("/Users/yyq/Desktop/300test 2.txt");
    for (int i = 0; i < n; i++){
        infile >> input[i];
    }
    printf(" Number of inversions are %d \n", cinversion(input, n));  //output total number of inversion
    
    return 0;
}
