//
//  BubbleSort
//
//  Created by yyq on 15/10/23.
//

#include <iostream>
#include <fstream>
#include <time.h>
using namespace std;

const int set_size = 160;                           //setup data size

void showResult(int ary[]){                        //function for result in array to show
    for (int i = 0; i < set_size; i++){
        cout<<ary[i]<<" ";
    }
    cout<<endl;
    return;
}
void BubbleSort(int ary[]){
    int i, tmp, j;
    for ( i = 0; i < set_size; ++i)                //recursion for bubble sort
        for ( j = 0; j < set_size - i - 1; ++j)    // j is no need to be same as i, just the one after i. because
                                                   // it will be covered when i from i to set_size
            if (ary[j] > ary[j + 1])               //if the elemnt before is bigger than the one after
            {
                tmp = ary[j];                      // swap these two
                ary[j] = ary[j + 1];
                ary[j + 1] = tmp;
            }
    showResult(ary);
}
int main(){
    double start,finish;                           // define and record start time
    start=(double)clock();
    
    cout<<"The input is "<<endl;                   // get data from file by using file handling, input only
    int input[set_size];
    ifstream infile;
    infile.open("/Users/yyq/Desktop/160test 1.txt");  //define where to find the file
    
    for (int i = 0; i < set_size; i++){              //put data into array
        infile >> input[i];
    }
    showResult(input);                               // recall showresult function
    BubbleSort(input);                               // recal bubble sort function
    
    finish=(double)clock();                          // record end time
    printf("%.2fms\n",finish-start);
    
    return 0;
}


