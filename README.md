# NICEMC playground

This project is a re-write of original [NICEMC program](https://github.com/ermongroup/a-nice-mc).

## Current Status

HMC and MH works fine.

NICE-MC seems running fine, acceptance ratio can be as high as original at Ring2D model.

## How to 

examples can be found in testscript folder.

Run `testscript/Normalsampler.py` to seem the result of HMC and MH of Ring2D model, in the file change `zSize` to corresponding  length(easy way to do it:  `zSize = energyFn.z.get_shape().as_list()[1] ` ). And let `energyFn` equals the model to evaluate. In default, the model is Ring2D and corresponding `zSize` is 2. And the result should be about 0 for mean and 1.456 for std.

Run `testscript/NICEtrain.py` to train NICE network, it will not load parameter from previous saving and run for 100'000 iterations by default, and it will save every 1'000 iterations. 

After training, run `testscript/NICEsampler.py` to sample using NICE network, in the file change `zSize` to corresponding  dimensions. And let `energyFn` equals the model to evaluate. In default, the model is Ring2D and corresponding `zSize` is 2. 

## Result

For now, after 200'000 iterations of training of NICE-MC, sampling 800 samples of Ring2D model with a batch size of 100, drop first 300 samples. HMC, MH and NICE-MC yields:

| Algor.       | mean       | std     | accept. | autocor. t       |
| :----------- | ---------- | ------- | ------- | ---------------- |
| MH           | 0.00859793 | 1.44885 | 0.33776 | 0.780032907052   |
| HMC          | 0.00535887 | 1.45801 | 0.9919  | 0.659679205674   |
| NICE-MC      | -0.0028535 | 1.454   | 0.69926 | -0.115518120 (?) |
| Ground truth | 0          | 1.456   | NA      | NA               |



