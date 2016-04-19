#!/usr/bin/python2.7

'''
Created on Mar 2, 2016

@author: lyulan
'''

import os
import sys, getopt
from sympy.functions.elementary.complexes import arg
import readprofile

def main(argv):
    inputfilefolder = ''
    outputfilefoler = ''
    try:
        opts, args = getopt.getopt(argv,"hi:o:",["ifile=","ofile="])
    except getopt.GetoptError:
        print 'tcss555 -i <inputfilefolder> -o <outputfilefolder>'
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print 'tcss555 -i <inputfilefolder> -o <outputfilefolder>'
            sys.exit()
        elif opt in ("-i", "--ifile"):
            inputfilefolder = arg
        elif opt in ("-o", "--ofile"):
            outputfilefoler = arg
    print 'Input file folder is: ', inputfilefolder
    if(os.path.isdir(inputfilefolder) == False):
        print 'Input folder does not exist'
        sys.exit(2)
    print 'Output file folder is: ', outputfilefoler
    
    readprofile.readprofile(inputfilefolder, outputfilefoler)

if __name__ == "__main__":
    main(sys.argv[1:])
    
    
    
    
   
