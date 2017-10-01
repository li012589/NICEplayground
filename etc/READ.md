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
3. Use tf.contrib.layers's fully_connected instead of utils/mlp.py still yielded acceptance ratio below 60%(~48%)
4. At least, sample without training seems right (using testNicemc.py without loading).
5. Add more test to NICE/nicelayer.py, now takes Jacobian into account.
6. Debug nice layer, the bias was added outside of active function. After debug, and after 5'000 iterations, acceptance ratio can high as ~60%.


### TODO

1. Train it ti see if it can perferm as well as original.
2. Benchmark

## 2017. 9. 19 Tue.

### Done

1. Train NICEMC/NICEMC.py for 200'000 iterations, acceptance ratio can be up to 73%, Train original NICE-MC the acceptance ratio can be up to about 75%. 

2. After training, run testNicemc.py, the result is:

   autoCorrelation:  -0.111961132942 acceptRate:  0.70292
   mean: -0.0127335
   std: 1.4507
   mean: -0.0150352
   std: 1.46035 

3. Debug: save parameter at end of all iterations.

### TODO

1. separate test to a single file as training script.
2. add matplotlib to plot training

## 2017. 9. 20 Thu. 

### Done

1. Test phi4 model, original yield maximum acceptance ratio of ~30% for training about 100'000 iterations. 
2. Clear code for easy read. Move all test script into test script folder.
3. Test phi4 model, my implement yield maximum acceptance ratio of ~12% for 10'000 iterations.

