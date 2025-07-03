该项目实现了对 《Simpler is More: Efficient Top-K Nearest Neighbors Search on Large Road Networks（PVLDB，2024）》 论文中提出的算法的复现。

原作者实现的版本见 [这里](https://github.com/DMA-Lab/kNN-Index-original)。
文章中提出的算法在大规模道路网络上进行高效的 Top-K 最近邻 fetch，通过巧妙地预计算所有顶点的 kNN，实现快速道路网络 k 近邻查询。
但该方法不允许查询时的 k 值大于索引构建时的 k 值。

本项目中， simple-solution 实现针对此的扩展。其中 `precomputition.h` 中实现了一个暴力的 kNN 计算，由于该方法是并行的，速度也不错。
`simple.cpp` 实现具体的查询策略。而 `main.cpp` 包含了一些遗留代码，可以不理会。