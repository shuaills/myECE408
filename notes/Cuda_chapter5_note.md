# Summary of CUDA Performance Optimization

## Global Memory Access

- Improve global memory bandwidth utilization via access coalescing  
  - Accesses from threads in a warp to consecutive addresses can be coalesced into one access
  - E.g. column-wise access in matrices can be coalesced, row-wise not
  - Matrix multiplication example: traversing N can be coalesced, not M 
- Combining shared memory tiling and access coalescing can significantly improve performance
  - Reduce global memory traffic via improved data reuse
  - E.g. divide M and N matrices into tiles loaded into shared memory  
  - Threads cooperate to load tiles in a coalesced pattern

## Memory Parallelism

- Understanding multi-channel and multi-bank DRAM system helps write efficient code
  - GPU has multiple channels, each connects to multiple banks
  - Banks can serve access requests in parallel
- Spreading thread accesses across channels and banks enables true parallel access
  - Interleave data distribution across channels and banks
  - E.g. matrix multiplication, partitioned access improves bank parallelism 
- Bank conflicts degrade efficiency, need optimization to avoid
  - Simultaneous accesses to one bank serialize 

## Warps and SIMD Execution

- Warps and SIMD significantly reduce hardware cost and improve pipeline efficiency
  - Threads in a warp controlled by one instruction unit
  - One instruction controls multiple data paths of threads
- Branch divergence decreases performance, avoid when possible
  - If-else clauses lead to divergent paths for threads in a warp 
  - Hardware execures different paths multiple times, reducing performance
  - E.g. conditionals or loop counts depending on thread ID
- Optimization: merge control paths, avoid access patterns depending on thread ID

## Dynamic Resource Partitioning

- CUDA dynamically partitions resources like registers, shared memory, etc
  - Allocate according to need improves utilization efficiency 
- Resources constrain each other, need to balance usage
  - Excessive shared memory reduces number of blocks
  - Compute occupancy to evaluate different strategies
- Optimization goal is maximizing occupancy, balancing different resources

## Thread Granularity

- Choose granularity according to algorithm pattern, balancing parallelism and efficiency
  - Coarse-grain threads process more data, fewer threads
  - Fine-grain threads do less work per thread, more threads
- Merging threads can reduce redundancy and improve efficiency
  - E.g. redundancy between neighbor blocks in matrix multiplication
  - Using larger tiles to merge threads improves efficiency  
- Need to balance parallelism vs efficiency tradeoff

## Summary

- Access coalescing, avoiding branch divergence, balancing resource occupancy, and optimizing thread granularity are key techniques for CUDA performance optimization.


# CUDA性能优化总结

## 全局内存访问

- 利用访问合并(coalescing)提高全局内存带宽利用率
  - 一个warp中的线程访问连续的全局内存地址可以合并为一个访问
  - 例如矩阵中的列优先访问可以合并,行优先访问不可以
  - 矩阵乘法示例:遍历N可以合并,遍历M不可以  
- 共享内存tiling结合访问合并可以大幅提升性能
  - 提高数据复用从而减少全局内存访问量
  - 例如将M和N矩阵划分为tile加载到共享内存
  - 线程协作加载tile实现访问合并

## 内存并行化

- 了解DRAM系统的多channel和多bank可以帮助编写高效代码
  - GPU有多个channel,每个channel连接多个bank
  - bank可以并行服务访问请求  
- 线程访问分散到不同的channel和bank,可以实现真正的并行访问
  - 交织分布数据到不同channel和bank
  - 例如矩阵乘法,分块分散访问改善bank并行度
- bank conflict会降低效率,需要优化避免
  - 多个线程同时访问一个bank会串行化 

## Warp和SIMD执行

- warp和SIMD可大幅降低硬件成本和提高管线效率
  - 一个warp中的线程由一个处理单元控制
  - 单指令控制多个线程的数据路径  
- 分支散射会导致性能下降,需要注意避免
  - if-else条件导致一个warp中的线程执行路径分散
  - 硬件多次执行不同路径,降低性能
  - 例如条件判断或循环计数依赖线程ID
- 优化方法:合并控制路径,避免访问模式依赖线程ID

## 动态资源分区

- CUDA动态分区资源如寄存器、共享内存等
  - 根据需要分配资源,提高利用效率
- 不同资源之间存在互相制约,需要平衡使用数量
  - 使用过多共享内存会减少块数量
  - 计算occupancy评估不同优化策略  
- 优化目标是最大化占用率,平衡不同资源

## 线程粒度

- 根据算法模式选择粒度,平衡并行度和计算效率
  - 粗粒度线程处理更多数据,线程数减少
  - 细粒度线程每个线程做少量工作,线程数增多
- 合并线程可以减少冗余计算提高效率 
  - 例如矩阵乘法中相邻线程块间的冗余
  - 使用更大的tile合并线程提升效率
- 需要权衡粒度带来的并行度和计算效率之间的权衡

## 总结

- 全局访问合并,避免分支散射,平衡资源占用率,优化线程粒度是优化CUDA性能的关键技术。