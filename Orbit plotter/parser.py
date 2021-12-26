import numpy as np
import pandas as pd
import csv

def expand(arr):
    arr=arr.copy()
    lens = list(map(lambda x: len(x),arr))
    print(lens)
    if(max(lens) != min(lens)):
        for i,j in enumerate(arr):
            if lens[i]<max(lens):
                toAdd = arr[i][len(arr[i])-1]
                for x in range((max(lens) - lens[i])):
                    arr[i].append(toAdd)
    return arr

def calculate_speeds(arr):
    toReturn =[]
    for i in range(1,len(arr[0])):
        speed=[]
        total_time = (arr[3][i]-arr[3][i-1])
        vector_speed=[]
        for j in range(3):
            vc = ((arr[j][i]-arr[j][i-1]))
            vector_speed.append(vc/total_time)
            speed.append(vc)
        speed = np.array(list(map(lambda x: np.power(x,2),speed)))
        speed = np.sqrt((np.sum(speed)))
        speed /= total_time
        vector_speed.append(speed)
        toReturn.append(vector_speed)
    
    return toReturn


# if(__name__=='main'):
def main():
    data = open('data.csv','r')
    # print(data)
    csv_reader = csv.reader(data, delimiter=',')
    line_count = 0
    for row in csv_reader:
        if line_count == 0:
            print(f'Column names are {", ".join(row)}')
        else:
            if row == []:
                continue
            print('*'*5,f'ROW {line_count}','*'*5)
            x,y,z,t=row[0].split(),row[1].split(),row[2].split(),row[3].split()            
            x,y,z,t=expand([x,y,z,t])
            x,y,z,t=np.array(x).astype(np.float32),np.array(y).astype(np.float32),np.array(z).astype(np.float32),np.array(t).astype(np.float32)

            # print(f'x:{x}\t\t\tx size:{len(x)}')
            # print(f'y:{y}\t\t\ty size:{len(y)}')
            # print(f'z:{z}\t\t\tz size:{len(z)}')
            # print(f't:{t}\t\t\tt size:{len(t)}')
            # speeds=calculate_speeds([x,y,z,t])
            # print('\nspeeds:')
            # print(pd.DataFrame(speeds,columns=['x','y','z','total'],index=t[1:]))
            print(pd.DataFrame(np.array([x,y,z]),columns=['x','y','z'],index=t))
        line_count += 1
    print(f'Processed {line_count} lines.')

main()