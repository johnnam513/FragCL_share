with open('log.txt', 'r') as f:
    with open('log2.txt', 'w') as g:
        for line in f.readlines():
            if 'Acc1' in line:
                g.write(line.split()[3] + '\n')


