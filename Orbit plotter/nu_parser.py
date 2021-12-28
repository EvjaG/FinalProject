import numpy as np
import pandas as pd
import csv

# if(__name__=='main'):
def main():
    data = open('data.csv','r')
    # print(data)
    csv_reader = csv.reader(data, delimiter=',')
    line_count = 0
    for row in csv_reader:
        if line_count == 0:
            row[0]=row[0].split()[1]
            print(f'Column names are {", ".join(row)}')
        else:
            if row == []:
                continue
            print('*'*5,f'ROW {line_count}','*'*5)
            t,x,y,z=row[0],row[1],row[2],row[3]        
            
            print(f"Time: {t}\tX: {x}\tY: {y}\tZ: {z}\t")

            
            # speeds=calculate_speeds([x,y,z,t])
            # print('\nspeeds:')
            # print(pd.DataFrame(speeds,columns=['x','y','z','total'],index=t[1:]))
            # print(pd.DataFrame(np.array([x,y,z]),columns=['x','y','z'],index=t))
        line_count += 1
    print(f'Processed {line_count-1} lines.')

main()