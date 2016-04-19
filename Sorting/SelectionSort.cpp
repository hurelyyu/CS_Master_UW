//
//  selectionsort.cpp
//

#include <iostream>
#include <fstream>
#include <time.h>
using namespace std;


const int set_size = 160;                            //define datasize

void showResult(int ary[]){                          //function for result in array to show
    for (int i = 0; i < set_size; i++){
        cout<<ary[i]<<" ";
    }
    cout<<endl;
    return;
}
void selectionSort(int ary[]){                       //for ary[i], find the smallest in the remaining entry, set
    int j, i;                                        //as min, swap a[i] and a[min]
    for(i = 0; i < set_size; i++){
        int min=i;
        for(j=i+1; j< set_size; j++){
            if(ary[j] < ary[min]){
                min = ary[j];
                min=j;
            }
        }
        int tmp = ary[min];
        ary[min]=ary[i];
        ary[i] = tmp;
        
        showResult(ary);                              //recall function to show result
    }
}
int main(){
    double start,finish;                              // define and record start time
    start=(double)clock();
    
    int input[set_size];                              //recall data size
    
    ifstream infile;                                  // get data from file by using file handling, input only
    infile.open("/Users/yyq/Desktop/160test 1.txt");  //define where to find the file
    
    for (int i = 0; i < set_size; i++){               //put data into array
        infile >> input[i];
    }
    showResult(input);
    selectionSort(input);                             // recal insertion sort function
    
    finish=(double)clock();                           // record end time
    printf("%.2fms\n",finish-start);
  
    return 0;
}