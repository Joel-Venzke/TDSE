import matplotlib.pyplot as plt
import numpy as np

data = []
with open("Prop_time.dat", 'r') as f:
  for line in f:
    tmp = []
    for i in line.strip().split(", "):
      tmp.append(float(i))
    data.append(tmp)
data =  np.array(data)

labels = ["tfqmr bjacobi jacobi","tfqmr bjacobi sor","tfqmr bjacobi ilu 0","tfqmr bjacobi ilu 1","tfqmr bjacobi ilu 2","tfqmr asm jacobi","tfqmr asm sor","tfqmr asm ilu 0","tfqmr asm ilu 1","tfqmr asm ilu 2","gmres bjacobi jacobi","gmres bjacobi sor","gmres bjacobi ilu 0","gmres bjacobi ilu 1","gmres bjacobi ilu 2","gmres asm jacobi","gmres asm sor","gmres asm ilu 0","gmres asm ilu 1","gmres asm ilu 2","bcgs bjacobi jacobi","bcgs bjacobi sor","bcgs bjacobi ilu 0","bcgs bjacobi ilu 1","bcgs bjacobi ilu 2","bcgs asm jacobi","bcgs asm sor","bcgs asm ilu 0","bcgs asm ilu 1","bcgs asm ilu 2"]
fig = plt.figure(figsize=(20,10))
ax = fig.add_axes([0.1, 0.1, 0.5, 0.75])
for line, lab in zip(data,labels):
  ax.plot([1,2,4],line,label=lab)

ax.set_xlabel('Processors', size=22)
ax.set_ylabel('Runtime (s)', size=22)
ax.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0., ncol=2,)
# plt.tight_layout(bbox_extra_artists=(lgd,))
plt.savefig('plot.png')
# plt.show()