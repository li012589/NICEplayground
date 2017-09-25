# NICEMC playground

This project is a re-write of original [NICEMC program](https://github.com/ermongroup/a-nice-mc).

## Current Status

HMC and MH works fine.

NICE-MC seems running fine, acceptance ratio can be as high as original at Ring2D model.

## How to 

examples can be found in *testscript* folder.

Run `testscript/Normalsampler.py` to seem the result of HMC and MH of Ring2D model, in the file change `zSize` to corresponding  length(easy way to do it:  `zSize = energyFn.z.get_shape().as_list()[1] ` ). And let `energyFn` equals the model to evaluate. In default, the model is Ring2D and corresponding `zSize` is 2. And the result should be about 0 for mean and 1.456 for std.

Run `testscript/NICEtrain.py` to train NICE network, it will not load parameter from previous saving and run for 100'000 iterations by default, and it will save every 1'000 iterations. 

After training, run `testscript/NICEsampler.py` to sample using NICE network, in the file change `zSize` to corresponding  dimensions. And let `energyFn` equals the model to evaluate. In default, the model is Ring2D and corresponding `zSize` is 2. 

**Remember to run all script in root directory of project**

## GAN computational graph



## NICE network 



## Folder Structure



## Saving Structure

### demo



## Result

### Ring2D 

After 200'000 iterations of training of NICE-MC, sampling 800 samples of Ring2D model with a batch size of 100, drop first 300 samples. HMC, MH and NICE-MC yields:

| Algor.       | mean        | std     | accept. | autocor. t     |
| :----------- | ----------- | ------- | ------- | -------------- |
| MH           | 0.00358582  | 1.45522 | 0.3388  | 13.9674        |
| HMC          | -0.00969827 | 1.45031 | 0.9923  | 6.46388        |
| NICE-MC      | 0.00226861  | 1.45189 | 0.6668  | -0.0995518 (?) |
| Ground truth | 0           | 1.456   | NA      | NA             |

### Phi4

Evaluating $ 3\times3, \kappa=1, \beta = 1 $ Ph4. 

After 200'000 iterations of training of NICE-MC, sampling 800 samples of Phi4 model with a batch size of 100, drop first 300 samples. HMC, MH and NICE-MC yields:

| Algor.  | mean       | std     | accept. | autocor. t |
| :------ | ---------- | ------- | ------- | ---------- |
| MH      | 0.164351   | 1.96883 | 0.00642 | 12.791     |
| HMC     | -0.421223  | 2.07029 | 0.8738  | 1.68466    |
| NICE-MC | -0.0109525 | 2.11119 | 0.33646 | 0.893479   |