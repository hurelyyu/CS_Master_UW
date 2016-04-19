//
// insertionSort.cpp
// insertionsort
//  Created by yyq on 15/10/22.
//
#include <iostream>
#include <fstream>
#include <time.h>

using namespace std;

const int set_size = 160;                      //define datasize

void showResult(int ary[]){                    //function for result in array to show
    for (int i = 0; i < set_size; i++){
        cout<<ary[i]<<" ";
    }
    cout<<endl;
    return;
}

void insertionSort(int ary[]){
    int tmp, j;
    for (int i = 1; i < set_size; i++){        //recursion for insertion sort
        tmp = ary[i];
        j = i;
        while (j > 0 && ary[j-1] > tmp){       //swap ary[i] with each larger interger on it left
            ary[j] = ary[j-1];
            j = j - 1;
        }
        ary[j] = tmp;
        showResult(ary);                       //recall function to show result
    }
}

int main(){
    double start,finish;                       // define and record start time
    start=(double)clock();
    
    int input[set_size];                       //recall data size
    
    ifstream infile;                           // get data from file by using file handling, input only
    infile.open("/Users/yyq/Desktop/160test 1.txt"); //define where to find the file
    
    for (int i = 0; i < set_size; i++){         //put data into array
     infile >> input[i];
    }
    showResult(input);
    insertionSort(input);                       // recal insertion sort function
    
    finish=(double)clock();                     // record end time
    printf("%.2fms\n",finish-start);
    
    return 0;
}
