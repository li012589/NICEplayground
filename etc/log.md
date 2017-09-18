## 2017. 9. 17 Sun.

### Done 

1. Debug NICE/nicelayer.py, add "reversed" at class NiceNetwork/backward.
2. Train NICEMC, run NICEMC/NICEME.py for about 10'000 iterations, the acceptance ratio maximum at about 47%, as contrast, the original program's acceptance can be about 60%~70%.

### TODO

1. Find if the acceptance ratio difference is casued by bad initialization. 
2. Keep train to see if acceptance ratio can go higher
3. Do some work about benchmark

## 2017. 9. 18 Mon.

### Done

1. Train NICEME, run NICEME/NICEMC.py for 200'000 iterations, acceptance ratio still stuck at 48% at most.
2. Using my NICEME/NICEMC.py and original nicelayer (a_nice_mc/utils/nice.py) is ok (60% acceptance ratio or higher), so maybe NICE/nicelayer still has bugs?
3.  Use tf.contrib.layers's fully_connected instead of utils/mlp.py still yielded acceptance ratio below 60%(~48%)
4. At least, sample without training seems right (using testNicemc.py without loading).
5. Add more test to NICE/nicelayer.py, now takes Jacobian into account.

 