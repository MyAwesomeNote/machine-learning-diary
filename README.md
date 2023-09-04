## Advanced Python(numpy, pandas, torch) for LLM study

### Day1 - 2023-09-04

> Already know per: 100%

- Git / GitHub usage
    - README.md
        - Markdown basic syntax
    - Edit file on GitHub
- Object oriented programming
    - Special method
    - Extend class and `super()`

#### Scala? Vector? Tensor?

- Scala [x]
- Vector [x, y]
- Tensor [x, y, ...z]

#### On GPU

```python
import torch

# !!! Before !!!
print(torch.cuda.is_available())  # It must be True

ex = torch.tensor([[1, 2], [3, 4]], device="cuda:0")  # cuda:n is index of GPU
res = ex.to("cpu").numpy()
print(res)
```

#### Controlling Shape

```python
import torch

a = torch.tensor([[1, 2, 3, 4], [5, 6, 7, 8]], dtype=torch.int8)
b = torch.tensor([[1, 2, 3, 4], [5, 6, 7, 8]], dtype=torch.int8)

c = a + b

print(c.shape)
print(c.view(8, 1))
print(c.view(1, 8))
```

