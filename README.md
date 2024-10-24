# Benchmark Various ANN Libraries

## Replicating
See [runbook](RUNBOOK.md). 

## Temp Results

   top k  recall (%)        var  mean (us)  p95 (us)  p99 (us)
0      1         100  27.413850     54.856      64.0      70.0
1      2         100  23.204355     53.479      61.0      66.0
2      5         100  21.494406     53.510      61.0      66.0
3     10         100  21.745870     53.628      62.0      66.0
4     20         100  17.962908     81.873      89.0      92.0
5     50         100  11.023377    127.260     132.0     135.0
6    100         100  10.161597    158.579     163.0     171.0
7    200         100  10.782377    193.471     199.0     203.0
8    500         100   5.084979    258.544     262.0     264.0
9   1000         100  11.226869    327.148     332.0     334.0

![output](output.png)