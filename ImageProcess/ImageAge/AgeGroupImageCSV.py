import pandas as pd
from PIL import Image
import numpy as np

# Age: A 18-24
# Age: B 25-34
# Age: C 35-49
# Age: D 50-xx

training_profile = '/Users/yaqunyu/UW_2016_Winter/TCSS555dataset/Train/Profile/Profile.csv'
image_folder = '/Users/yaqunyu/UW_2016_Winter/TCSS555dataset/Train/Image/'
df = pd.read_csv(training_profile,sep=',')
df2 = pd.DataFrame
tempdf = pd.DataFrame
matrix = [[]*10000 for x in xrange(9500)]
#the output csv file
fi = open("/Users/yaqunyu/Desktop/imagecsvs/ImageInfo.csv", "a")
ip=0
for index, row in df.iterrows():
    print ip
    userid = row['userid']
    age = row['age']
    gender = row['gender']
    ope = row['ope']
    con = row['con']
    ext = row['ext']
    agr = row['agr']
    neu = row['neu']
    #print userid
    im = Image.open(image_folder + userid + ".jpg").resize((100,100))
    im=im.convert('L') #makes it greyscale
    t=list(im.getdata())
    for i in range(len(t)):      #transfer to string value in order to put csv
        t[i] = str(t[i])
    matrix[index].append(userid)
    matrix[index]  += t
    matrix[index].append(str(gender))
    if age >= 18 and age <= 24:
        matrix[index].append(str('18-24')) 
    elif age >= 25 and age <= 34:
        matrix[index].append(str('25-34'))
    elif age >= 35 and age <= 49:
        matrix[index].append(str('35-49'))
    else:
        matrix[index].append(str('50-xx'))
    matrix[index].append(str(age)) 
    matrix[index].append(str(ope))
    matrix[index].append(str(con))
    matrix[index].append(str(ext))
    matrix[index].append(str(agr))
    matrix[index].append(str(neu)) 
    line = ",".join(matrix[index])
    fi.write(line)
    fi.write("\n")
    ip = ip +1

fi.close()



   





