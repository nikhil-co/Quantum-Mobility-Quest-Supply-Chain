import numpy as np
import random
import csv

def coefficients(Np=10,Ns=7):
    C=np.zeros((Np, Ns, Ns))
    f1=np.zeros((Np))
    f2=np.zeros((Ns,Ns))
    random.seed(1)
    for a in range (2,Np+1):
        f1[a-1]=random.randrange(50, 1000)/1000
    for i in range (1,Ns):
        for j in range (i+1,Ns+1):
            f2[i-1,j-1]=random.randrange(100, 1000)/100
    for a in range (2,Np+1):
        for i in range (1,Ns):
            for j in range (i+1,Ns+1):
                C[a-1,i-1,j-1]=round(f1[a-1]*f2[i-1,j-1],2)
#                print(a,i,j,f1[a-1],f2[i-1,j-1],C[a-1,i-1,j-1])
    header = ['Part', 'Departure/Arrival site', 'Arrival/Departure site', 'Cost']
    with open('cost_'+str(Np).zfill(2)+'_'+str(Ns).zfill(2)+'.csv', 'w') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for a in range (2,Np+1):
            for i in range (1,Ns):
                for j in range (i+1,Ns+1):
                    data=[a,i,j,C[a-1,i-1,j-1]]
                    writer.writerow(data)
    f.close()

Np = int(input("How many different parts in PBS? (default 10)").strip() or "10")
Ns = int(input("How many different sites? (default 7)").strip() or 7)
print("Number of parts =", Np)
print("Number of sites =", Ns)
coefficients(Np,Ns)
print('Cost coefficients written in file cost_'+str(Np).zfill(2)+'_'+str(Ns).zfill(2)+'.csv')