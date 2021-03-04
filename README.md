# Introduction

Implement the algorithm presented in [1](http://www.cse.cuhk.edu.hk/~leojia/projects/pencilsketch/pencil_drawing.htm) by CUDA. You can use the code like below to get a pencil drawing production:

```
./gpu_pencil_draw <input> <pencil image>
```

or run cpu version:

```
./cpu_pencil_draw <input> <pencil image>
```

# Result

- demo 1

![tree: original](https://github.com/Fibird/cuda_pencildraw/blob/master/inputs/input-01.jpg) 
![tree: pencil draw](https://github.com/Fibird/cuda_pencildraw/blob/master/results/result-01.png)

- demo 2

![tree: original](https://github.com/Fibird/cuda_pencildraw/blob/master/inputs/input-02.jpg) 
![tree: pencil draw](https://github.com/Fibird/cuda_pencildraw/blob/master/results/result-02.png)

# Reference

1. Lu C, Xu L, Jia J. Combining sketch and tone for pencil drawing production[C]//Proceedings of the Symposium on Non-Photorealistic Animation and Rendering. Eurographics Association, 2012: 65-73.
2. GitHub repo: [candycat1992/PencilDrawing](https://github.com/candycat1992/PencilDrawing)