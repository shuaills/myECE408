# Parallel Computation Pattern: Convolution

## 7.1 Background

### Definition of Convolution
- Convolution is an array operation that generates an output array by weighted summation of an input signal array using a template array (convolution kernel).
- The convolution kernel is also commonly referred to as the Mask. 
- The input data is also known as the feature data.

### Applications of Convolution
- 1D convolution is mainly used in signal processing, with 1D input and output arrays.
- 2D convolution is mainly used in image processing, with 2D input and output arrays.
- 3D convolution also has applications but is less common.

### Boundary Conditions in Convolution
- In convolution computation, output elements near boundaries need to access out-of-bound input elements.
- Different applications handle these out-of-bound elements differently. 
- Most common is assigning default values like 0. Some copy values from the boundary.
- This introduces the boundary condition problem that convolution must handle.

## 7.2 Basic Parallel Convolution Algorithm 

### Algorithm Idea
- Distribute threads according to output size, each thread computes one output element.
- Compute weighted sum by offset access of input array. 
- Use if statements to detect out-of-bound and apply default values.

### Optimization Challenges
- Redundant loading of input data, global memory bandwidth bottleneck.
- Thread divergence on boundaries.
- Low computation density, frequent memory access.

## 7.3 Using Constant Memory

### Properties of Constant Memory
- Read/write patterns suitable for storing Mask array.
- Limited size, but Mask arrays are usually small.

### Leveraging Constant Cache
- Allows repeated access without loading into registers.
- Broadcast mechanism provides efficient access to all threads.

### Optimization Effect  
- Reduces global memory accesses.
- Improves computation density.
- Doubles throughput.

## 7.4 Tiled Algorithm

### Algorithm Idea
- Divide input data into Tiles, loading into Shared Memory. 
- Reduces redundant accesses, improves reuse.
- Tiles include Halo Cells for use by boundary threads.

### Implementation Keys
- Distinguish Input and Output Tiles.
- Handle boundary loading conditions.
- Synchronized access to Shared Memory. 

### Optimization Effect
- Reduces global memory traffic by about Mask size times.
- Improves computation density.
- But also increases code complexity.

## 7.5 Tiled Algorithm Using L2 Cache

### Algorithm Idea
- Only load Tile center into Shared Memory.
- Directly access Halo Cells in global memory. 
- Rely on L2 Cache to automatically cache Halo Cells.

### Implementation Keys
- Reduces Shared Memory usage. 
- Access pattern is regular with locality.

### Optimization Effect
- Allows larger Tile size for more significant effect.
- But relies on cache hits, has risks.

## 7.6 2D Tiled Convolution 

### Algorithm Idea
- Similar to 1D tiled convolution.
- Handling 2D boundaries more complex.

### Implementation Keys  
- Calculate 2D thread coordinates.
- Load extended 2D Input Tile.
- Synchronized access to Shared Memory.

### Optimization Effect
- Reduces accesses by about (Mask size)^2 times.  
- Proper Tile size very important.


# 并行计算模式:卷积 

## 7.1 背景

### 卷积的定义
- 卷积是一种数组运算,通过用一个模板数组(卷积核)对一个信号数组进行加权求和来生成输出数组。
- 卷积核通常也称为Mask。
- 输入的数据通常称为特征数据。

### 卷积在不同领域的应用
- 1D卷积主要应用于信号处理,输入和输出都是1维数组。
- 2D卷积主要应用于图像处理,输入和输出都是2维数组。 
- 3D卷积也有应用,但较为少见。

### 卷积计算中的边界条件
- 在卷积计算中,输出边界附近的元素通常需要访问越界的输入元素。
- 不同应用中,对这些越界元素的处理方法也不同。
- 最常见的是赋予默认值,比如0。也有的应用会复制边界元素的值。
- 这引入了卷积计算必须处理的边界条件问题。

## 7.2 基本并行卷积算法

### 算法思路
- 将线程按输出元素数量分配,每个线程计算一个输出元素。
- 通过偏移访问输入数组,进行加权求和计算。
- 使用if语句检测越界,应用默认边界值。

### 算法优化难点
- 存在冗余的输入数据加载,全局内存带宽成为瓶颈。
- 边界线程存在控制流分支。
- 计算密度低,内存访问频繁。

## 7.3 使用Constant Memory

### Constant Memory特性
- 读写特性适合存放Mask数组。
- 大小有限,但Mask数组通常很小。 

### 利用Constant Cache
- 可以重复访问,不需要加载到寄存器。
- 广播机制可以高效提供给所有线程。

### 优化效果
- 减少全局内存访问。
- 提高计算密度。
- 倍增吞吐量。

## 7.4 分块算法

### 算法思路
- 将输入数据分为Tile,加载到Shared Memory。
- 降低冗余访问,提高重复利用。
- Tile包括Halo Cells,供边界线程使用。

### 实现要点
- 区分Input Tile和Output Tile。
- 处理加载边界条件。
- 同步访问Shared Memory。

### 优化效果
- 降低全局内存访问约Mask大小倍。 
- 提高计算密度。
- 但也增加了代码复杂度。

## 7.5 利用L2 Cache的分块算法

### 算法思路
- 只加载Tile中心部分到Shared Memory。
- 直接访问全局内存中的Halo Cells。
- 依赖L2 Cache自动缓存Halo Cells。

### 实现要点  
- 减少Shared Memory用量。
- 访问模式taggit规整,有局部性。

### 优化效果
- 可以使用更大Tile,优化效果更明显。
- 但依赖Cache命中,有一定风险。

## 7.6 2D分块卷积

### 算法思路
- 类似1D分块卷积。 
- 处理2D的边界条件更复杂。

### 实现要点
- 计算2D线程坐标。
- 加载扩展后的2D Input Tile。 
- 同步访问Shared Memory。

### 优化效果
- 减少访问约(Mask大小)平方倍。
- 合理Tile大小很重要。
