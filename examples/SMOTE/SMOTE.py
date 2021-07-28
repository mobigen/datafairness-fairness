import os.path
import sys
import pandas as pd

filepath = sys.argv[1] # 1st argument, which is absolute path of csv file
dirname = os.path.dirname(filepath)
filename = os.path.basename(filepath)

data = pd.read_csv(filepath, header = 0) # read_csv with header
Critical = float(sys.argv[2]) # 2nd argument, which is critical value

count = data.groupby(['gender']).count()
gender_count = [count.loc[0], count.loc[1]] # count and save the number of row for each gender

Major_gender_bit = gender_count[0] < gender_count[1] # if the number of 'gender 1' is larger, set 1
Major_n = int(gender_count[int(Major_gender_bit)])
Minor_n = int(gender_count[int(~Major_gender_bit)])
GB = Major_n / (Major_n + Minor_n)

if(GB > Critical): # Gender-Bias is larger than Critical
    repeat_n = round(Critical * Major_n - (1 - Critical) * Minor_n)
    if int(Major_gender_bit):
        upsample = data.query('gender == 0').sample(repeat_n, replace = True) # upsampling data 'gender' == 0 with replacement
        print("Upsampled with 'gender 0'")
    else : 
        upsample = data.query('gender == 1').sample(repeat_n, replace = True) # upsampling data 'gender' == 1 with replacement
        print("Upsampled with 'gender 1'")
    
    a = filename.split('.')
    filename = a[0] + '_upsampled.' + a[1] # original_name_upsample.csv
    filepath = os.path.join(dirname, filename)
    
    data = pd.concat([data, upsample], ignore_index = True)
    data.to_csv(filepath)
else :
    print("No need to sample")