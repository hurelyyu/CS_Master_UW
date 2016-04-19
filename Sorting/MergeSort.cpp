
//
//  MergeSort.cpp
//  Simulating Sort
//
//  Created by yyq on 15/10/23.
//  Copyright © 2015年 yyq. All rights reserved.
//

#include <iostream>
#include <fstream>
#include <time.h>
using namespace std;

const int set_size = 160;

void showResult(int ary[]){
    for (int i = 0; i < set_size; i++){
        cout<<ary[i]<<" ";
    }
    cout<<endl;
    return;
}

void merge(int a[], int l, int m, int h){
    int l_len = m - l + 1;
    int h_len = h - m;
    int low[l_len];
    int high[h_len];
    int i; //index for low[];
    int j; //index for high[];
    
    //copy to tmp arrays
    for (i = 0; i < l_len; i++){
        low[i] = a[i+l];
    }
    for (j = 0; j < h_len; j++){
        high[j] = a[j+m+1];
    }
    i = 0;
    j = 0;
    
    for(int k = l; k <= h; k++){
        if (i >= l_len){//when low is empty, copy the remainder in high to a[]
            a[k] = high[j];
            j = j + 1;
        }
        else if (j >= h_len){//when high is empty, copy the remainder in low to a[]
            a[k] = low[i];
            i = i + 1;
        }
        else {
            if (low[i] < high[j]){
                a[k] = low[i];
                i = i + 1;
            }
            else {
                a[k] = high[j];
                j = j + 1;
            }
        }
    }
    showResult(a);
}


void mergeSort(int ary[], int low, int high){
    if(low<high){
        int mid=(high-low)/2+low;
        mergeSort(ary,low, mid);
        mergeSort(ary, mid+1, high);
        merge(ary,low,mid,high);
    }
      //showResult(ary); to void duplicate
}

int main(){
    double start,finish;
    start=(double)clock();
    
    
    int input[set_size];
    
    ifstream infile;
    infile.open("/Users/yyq/Desktop/160test 1.txt");
    
    for (int i = 0; i < set_size; i++){
        infile >> input[i];
    }
    
    //showResult(input); this will only display input
    mergeSort(input, 0, set_size-1);
    
    finish=(double)clock();
    printf("%.2fms\n",finish-start);
    return 0;
}