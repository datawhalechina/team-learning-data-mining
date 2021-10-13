# ARMA 时间序列模型与预测

### 白噪声

可以通过 Box-Ljung 检验来检验序列是否为白噪声：

```R
> set.seed(100)
> data = rnorm(100)
> Box.test(data, type='Ljung', lag = log(length(data)))

	Box-Ljung test

data:  data
X-squared = 8.896, df = 4.6052, p-value = 0.09169
```

从结果中可以看见 $p = 0.09169 > 0.05$，因此无法拒绝序列为白噪声的假设。下面绘制一下该序列的图像以及 ACF 图像：

```R
> op <- par(mfrow=c(2, 1), mar=c(5, 4, 2, 2) + .1)
> plot(ts(data))
> acf(z, main = "")
> par(op)
```

![image-20210924205843412](.\typora-user-images\image-20210924205843412.png)

### AR(p) 序列

先给出一个 AR(1) 的手工计算例子，在这个例子中可以通过 $y_{n-1}$ 预测 $y_n$

```R
> n <- 200
> x <- rnorm(n)
> f = c(1,2)
> y1 = x[1]; y1
[1] 1
> y2 = x[2] + f[1] * y1; y2
[1] 3
> y3 = x[3] + f[1] * y2 + f[2] * y1; y3
[1] 8
> y4 = x[4] + f[1] * y3 + f[2] * y2; y4
[1] 18
```

实际上也可以通过 `filter()` 函数来计算：

```R
> x = 1:10; x
 [1]  1  2  3  4  5  6  7  8  9 10
> y = filter(x, c(1, 2), method='r'); y
Time Series:
Start = 1 
End = 10 
Frequency = 1 
 [1]    1    3    8   18   39   81  166  336  677 1359
```

下面计算一个 $p=3$ 的例子：

```R
> n <- 200
> x <- rnorm(n)
> f = c(.3, -.7, .5)
> y <- rep(0, n)
> y[1:3] = x[1:3]
> for (i in 4:n) {
+ 	y[i] <- f[1]*y[i-1] +f[2]*y[i-2] + f[3]*y[i-3] + x[i]
+ }
> op <- par(mfrow=c(3,1), mar=c(2,4,2,2)+.1)
> plot(ts(y), xlab="", ylab="AR(3)")
> acf(y, main="", xlab="")
> pacf(y, main="", xlab="")
> par(op)
```

![image-20210923195745948](.\typora-user-images\image-20210923195745948.png)

同样地，也可以通过 `filter()` 函数来计算：

```R
> y = filter(x, c(.3, -.7, .5), method='r'); y
> op <- par(mfrow=c(3,1), mar=c(2,4,2,2)+.1)
> plot(ts(y), xlab="", ylab="AR(3)")
> acf(y, main="", xlab="")
> pacf(y, main="", xlab="")
```

![image-20210923200033331](.\typora-user-images\image-20210923200033331.png)

结果是一样的。

### MA 模型

直接使用 `filter()` 函数计算 MA 模型：

```R
> x = 1:10
> y = filter(x, filter = c(.5, .3)); y # filter 函数未设置 method，默认为'c'，即使用滑动平均
Time Series:
Start = 1 
End = 10 
Frequency = 1 
 [1] 1.3 2.1 2.9 3.7 4.5 5.3 6.1 6.9 7.7  NA
```

作业：编写手工计算 MA 模型的代码

### ARIMA 模型相关

R 语言中自带的 `arima.sim()` 函数可以模拟生成 AR、MA、ARMA 或 ARIMA 模型的数据。其原型为：

```R
arima.sim(model, n, rand.gen = rnorm, innov = rand.gen(n, ...),
         n.start = NA, start.innov = rand.gen(n.start, ...),
         ...)
```

其中，`model` 是一个列表，用于指定各模型的系数；`order` 是 ARIMA(p, d, q) 中 $(p, d, q)$ 三个元素的向量，$p$ 为 AR 阶数， $q$ 是 MA 的阶数，$d$ 是差分阶数。例如，模拟如下的 ARIMA(1, 1, 1) 模型，并产生长度为300的样本：

$Y_t = X_t - X_{t-1}, X_t = -0.9X_{t-1} + \varepsilon_t + 0.5\varepsilon_{t-1}, \varepsilon_t \sim WN(0, 2^2)$

R 语言代码为：

```R
> x <- 2.0 * arima.sim(model = list(
+     ar = c(-0.9), ma = c(0.5), order = c(1, 1, 1)), n=300)
```



