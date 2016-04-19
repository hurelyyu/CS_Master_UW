//
//  main.cpp
//  QuickSort


#include <iostream>
#include <fstream>
#include <time.h>

using namespace std;

const int n = 40;
void swap(int* a, int* b)                    // A utility function to swap two elements
{
    int t = *a;
    *a = *b;
    *b = t;
}

/* take last element as pivot, places the pivot element at its
 correct position in sorted array, and places all smaller (smaller than pivot)
 to left of pivot and all greater elements to right of pivot */
int partition (int ary[], int low, int high)
{
    int x = ary[high];    // pivot
    int i = (low - 1);  // Index of smaller element
    
    for (int j = low; j <= high- 1; j++)
    {
        // If current element is smaller than or equal to pivot
        if (ary[j] <= x)
        {
            i++;    // increment index of smaller element
            swap(&ary[i], &ary[j]);  // Swap current element with index
        }
    }
    swap(&ary[i + 1], &ary[high]);
    return (i + 1);
}

// ary[] --> Array to be sorted, l  --> Starting index, h  --> Ending index
void quickSort(int ary[], int l, int h)
{
    if (l < h)
    {
        int p = partition(ary, l, h); // Partitioning index
        quickSort(ary, l, p - 1);
        quickSort(ary, p + 1, h);
    }
}

// Function to print an array
void showResult(int ary[]){
    for (int i = 0; i < n; i++){
        cout<<ary[i]<<" ";
    }
    cout<<endl;
    return;
}

// recall above functions
int main()
{
    double start,finish;           //record start time
    start=(double)clock();
    
    int ary[n];
    ifstream infile;
    infile.open("/Users/yyq/Desktop/160test 1.txt");
    
    for (int i = 0; i < n; i++){
        infile >> ary[i];}
    
    quickSort(ary, 0, n-1);
    showResult(ary);
    
    
    finish=(double)clock();      //record ending time
    printf("%.2fms\n",finish-start);
    
    
    cout<<"short:"<<sizeof(short)<<endl;   //record memory useage for each int, char, short, long, float, double
    cout<<"char:"<<sizeof(char)<<endl;
    cout<<"int:"<<sizeof(int)<<endl;
    cout<<"long:"<<sizeof(long)<<endl;
    cout<<"float:"<<sizeof(float)<<endl;
    cout<<"double:"<<sizeof(double)<<endl;
    
    return 0;
    
}
