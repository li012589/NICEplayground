# NICEMC playground

This project is a re-write of original [NICEMC program](https://github.com/ermongroup/a-nice-mc).

## Current Status

HMC and MH work fine.

NICE-MC may have some problems, but still yields right result for at least Ring2D model(with mean ~0, std ~1.456). And training does improve it's performance but acceptance ratio seemed stuck at 48% somehow(with original program at ~60%).

## How to 

Run `evl.py` to seem the result of HMC and MH of Ring2D model, in the file change `zDim` to corresponding  dimensions. And let `energyFn` equals the model to evaluate. In default, the model is Ring2D and corresponding `zDim` is 2. And the result should be about 0 for mean and 1.456 for std.

Run `NICEMC\NICEMC.py` to train NICE network, it will not load parameter from previous saving and run for 100'000 iterations by default, and it will save every 1'000 iterations. 

After training, run `testNicemc.py` to sample using NICE network, in the file change `zDim` to corresponding  dimensions. And let `energyFn` equals the model to evaluate. In default, the model is Ring2D and corresponding `zDim` is 2. 

## Result

For now, after 100'000 iterations of training, sampling 800 samples of Ring2D model with a batch size of 100, drop first 300 samples. HMC, MH and NICE-MC yielded:

| Algor.  | mean        | std     | accept. | autocor. t    |
| :------ | ----------- | ------- | ------- | ------------- |
| MH      | 0.0786284   | 1.43155 | NA      | 19.2251491761 |
| HMC     | -0.00902451 | 1.4578  | NA      | 5.53441307631 |
| NICE-MC | 0.00999073  | 1.45456 | 0.4605  | ~2            |



