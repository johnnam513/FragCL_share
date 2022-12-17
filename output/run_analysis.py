import sys
import numpy as np

datas = ['bbbp', 'tox21', 'toxcast', 'sider', 'clintox', 'muv', 'hiv', 'bace']
seeds = [0,1,2]
data_str = ''
for seed in seeds:
    #data_str = ''
    for data in datas:
        with open(f"{sys.argv[1]}/{seed}/{data}.out") as f:
            for line in f.readlines():
                if "best train:" in line:
                    data_str = data_str + line.split()[-1] + ' '
    data_str += '\n'
print(data_str)


num = 3
datas = np.zeros((num, len(datas)))
for (i,line) in enumerate(data_str.split("\n")):
    for (j,val) in enumerate(line.split()):
        datas[i,j] = float(val)

#print(datas)
mean = list(np.mean(datas, axis=0)*100)
std = list(np.std(datas, axis=0, ddof=1)*100)
maximum = list(np.max(datas, axis=0)*100)

table_str = '& '
for i in range(len(mean)):
    table_str += str(round(mean[i], 1)) + "\stdv{" + str(round(std[i],1)) + "}" + ' & '

table_str += str(round(np.mean(np.mean(datas, axis=0), axis=0)*100,2))
max_str = ''
for i in range(len(maximum)):
    max_str += str(round(maximum[i], 1)) + " & "

print(table_str)
print(max_str)
