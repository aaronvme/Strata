# Decision Tree Splitting & MDI Mechanics

This document details how Strata computes recursive binary partition splits and Mean Decrease in Impurity (MDI) feature importances with $O(1)$ streaming histogram updates.


---

## 1. Streaming Histogram Split Search

Given a continuous feature vector $\mathbf{x} \in \mathbb{R}^N$ and target labels $\mathbf{y}$, finding the optimal threshold $s$ requires evaluating impurity for all candidate split points.

Naive split evaluation recalculates class counts in $\mathcal{O}(N)$ for each candidate threshold, yielding $\mathcal{O}(N^2)$ complexity per node.

### Strata's $\mathcal{O}(N \log N)$ Streaming Engine:
1. Samples are sorted by feature values: $\mathbf{x}_{(1)} \le \mathbf{x}_{(2)} \le \dots \le \mathbf{x}_{(N)}$.
2. A single pass streams counts from the **Right Partition** into the **Left Partition**:
   $$\text{LeftCount}_c \leftarrow \text{LeftCount}_c + 1, \quad \text{RightCount}_c \leftarrow \text{RightCount}_c - 1$$
3. Gini impurity or MSE is updated in $\mathcal{O}(K)$ time for each candidate split point:
   $$I_{\text{Gini}}(L) = 1 - \sum_{c=1}^K \left(\frac{\text{LeftCount}_c}{N_L}\right)^2$$

---

## 2. Flat-Buffer Tree Layout

Instead of pointer-chasing tree structures where each node is a separate heap allocation, Strata stores tree topologies in contiguous flat array buffers:

```mojo
struct Node(Copyable, Movable):
    var feature: Int        # Split feature index (-1 for leaves)
    var threshold: Float64  # Split threshold
    var left: Int           # Index into nodes buffer
    var right: Int          # Index into nodes buffer
    var impurity: Float64   # Impurity at this node
    var n_node_samples: Int # Sample count
```

### Advantages:
- **Cache-Friendly Inference**: Evaluating predictions streams through sequential buffer indices.
- **Zero-Fragmentation Garbage Collection**: Destroying a tree is a single buffer deallocation.
