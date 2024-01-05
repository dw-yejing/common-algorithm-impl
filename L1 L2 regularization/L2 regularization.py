import numpy as np
from sklearn.datasets import make_regression

# 生成一个样本量为100，特征数为10的数据集
X, y = make_regression(n_samples=100, n_features=10, random_state=42)

# 定义L2正则化参数λ
lambd = 0.1

# 定义损失函数
def loss(X, y, w, lambd):
    y_pred = X.dot(w)
    error = y - y_pred
    return np.mean(error ** 2) + lambd * np.sum(w ** 2)

# 定义训练函数
def train(X, y, learning_rate, num_epochs, lambd):
    # 初始化权重向量
    w = np.random.rand(X.shape[1])
    
    for epoch in range(num_epochs):
        # 计算梯度
        y_pred = X.dot(w)
        error = y - y_pred
        grad = -2 * X.T.dot(error) + 2 * lambd * w
        
        # 更新权重
        w -= learning_rate * grad
        
        # 计算损失
        current_loss = loss(X, y, w, lambd)
        
        # 打印损失
        if epoch % 100 == 0:
            print("Epoch:", epoch, "Loss:", current_loss)
    
    return w

# 训练模型
w = train(X, y, learning_rate=0.1, num_epochs=1000, lambd=lambd)

# 在测试集上进行预测
X_test = np.random.rand(20, 10)
y_pred = X_test.dot(w)
print("预测值:", y_pred)
