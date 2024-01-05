import numpy as np

# 生成一个样本量为100，特征数为10的数据集
X = np.random.rand(100, 10)
y = np.random.rand(100)

# 定义L1正则化参数λ
lambd = 0.1

# 初始化权重向量
w = np.random.rand(X.shape[1])

# 定义模型损失函数
def loss(X, y, w, lambd):
    y_pred = X.dot(w)
    error = y - y_pred
    return np.mean(error ** 2) + lambd * np.sum(np.abs(w))

# 定义模型训练函数
def train(X, y, w, lambd, learning_rate, num_epochs):
    for epoch in range(num_epochs):
        # 计算梯度
        y_pred = X.dot(w)
        error = y - y_pred
        grad = -2 * X.T.dot(error) + lambd * np.sign(w)
        
        # 更新权重
        w -= learning_rate * grad
        
        # 计算损失
        current_loss = loss(X, y, w, lambd)
        
        # 打印损失
        if epoch % 100 == 0:
            print("Epoch:", epoch, "Loss:", current_loss)
    
    return w

# 训练模型
w = train(X, y, w, lambd, learning_rate=0.1, num_epochs=1000)

# 在测试集上进行预测
X_test = np.random.rand(20, 10)
y_pred = X_test.dot(w)
print("预测值:", y_pred)
