# 数値解析学演習課題
数値解析学にて出された演習課題のプログラム

## NN_fashion_mnist.py
fashion_mnistについてCNNを利用した学習を行い結果をテキストデータとして出力する  
### CNNの層について
CNNの層は下記の様になっている
```
CNN(
  (head): Sequential(
    (0): Conv2d(1, 32, kernel_size=(3, 3), stride=(1, 1))
    (1): ReLU()
    (2): MaxPool2d(kernel_size=1, stride=1, padding=0, dilation=1, ceil_mode=False)
    (3): Conv2d(32, 32, kernel_size=(3, 3), stride=(1, 1))
    (4): ReLU()
    (5): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
    (6): Conv2d(32, 64, kernel_size=(3, 3), stride=(1, 1))
    (7): ReLU()
    (8): MaxPool2d(kernel_size=1, stride=1, padding=0, dilation=1, ceil_mode=False)
    (9): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1))
    (10): ReLU()
    (11): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
  )
  (tail): Sequential(
    (0): Linear(in_features=1024, out_features=516, bias=True)
    (1): ReLU()
    (2): Linear(in_features=516, out_features=320, bias=True)
    (3): ReLU()
    (4): Linear(in_features=320, out_features=228, bias=True)
    (5): ReLU()
    (6): Linear(in_features=228, out_features=120, bias=True)
    (7): ReLU()
    (8): Linear(in_features=120, out_features=80, bias=True)
    (9): ReLU()
    (10): Linear(in_features=80, out_features=20, bias=True)
    (11): ReLU()
    (12): Linear(in_features=20, out_features=10, bias=True)
  )
)
```

## NN_fashion_mnist_preview.py
fashion_mnistにて出力されたテキストデータからグラフを表示する
