'''
Created on Mar 2, 2016

@author: lyulan
'''

import os

import ml


def readprofile(testdata_file_path, output_file_path):
    
    #if the output file path does not exit, build it
    if(os.path.isdir(output_file_path) == False):
        os.makedirs(output_file_path)
        pass
 
    #use ML algorithm to get the 5 personality predictions
    result_df = ml.predict_personality(testdata_file_path);
    
    for row in result_df.itertuples():
        print(row)
        userid = row[1]
        output_file = open(output_file_path + "/" + userid + ".xml", 'w')
        output_file.write("<userId = \"{" + userid + "}\"\n"
                          "age_group = \"\"\n"
                          "gender = \"\"\n"
                          "extrovert = \"" + str(row[4]) + "\"\n"
                          "neurotic = \"" + str(row[6]) + "\"\n"
                          "agreeable = \"" + str(row[5]) + "\"\n"
                          "conscientious = \"" + str(row[3]) + "\"\n"
                          "open = \"" + str(row[2]) + "\"\n"
                          "/>")
        output_file.close()
        del output_file
        pass
    
    
    
        

