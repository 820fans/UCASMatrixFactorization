import numpy
from pylab import *
import codecs


def load_data(file_path):
    f = open(file_path)
    V = []
    for line in f.readlines():
        lines = line.strip().split(",")
        data = []
        for x in lines:
            data.append(float(x))
        V.append(data)
    return mat(V)


def train(V, r, k, e):
    m, n = shape(V)
    W = mat(np.random.rand(m, r))
    H = mat(np.random.rand(r, n))

    f_out = codecs.open('result_nmf', 'w', 'utf-8')

    for x in range(k):
        # error
        V_pre = W * H
        E = V - V_pre
        # print E
        err = 0.0
        for i in range(m):
            for j in range(n):
                err += E[i, j] * E[i, j]
        # print('error rate %f' % err)
        f_out.write(str(err) + '\r\n')
        if err < e:
            break

        a = W.T * V
        b = W.T * W * H
        for i_1 in range(r):
            for j_1 in range(n):
                if b[i_1, j_1] != 0:
                    # 进行迭代
                    H[i_1, j_1] = H[i_1, j_1] * a[i_1, j_1] / b[i_1, j_1]

        c = V * H.T
        d = W * H * H.T
        for i_2 in range(m):
            for j_2 in range(r):
                if d[i_2, j_2] != 0:
                    W[i_2, j_2] = W[i_2, j_2] * c[i_2, j_2] / d[i_2, j_2]

    f_out.close()
    return W, H


def draw():
    data = []

    f = open("result_nmf")
    for line in f.readlines():
        lines = line.strip()
        data.append(lines)

    n = len(data)
    x = range(n)
    plot(x, data, color='r', linewidth=3)
    plt.title('收敛曲线')
    plt.xlabel('迭代伦次')
    plt.ylabel('损失')
    show()


def testRun():
    file_path = "./data1"

    V = load_data(file_path)
    W, H = train(V, 2, 100, 1e-7)

    print("原始矩阵")
    print(V)
    print("分解后WH-E")
    print(W * H)
    print("分解矩阵")
    print(W)
    print(H)

    draw()


if __name__ == "__main__":
    B = numpy.mat([[1, 4, 0, -1]]).T
    U = numpy.mat([[-2, 1, 3, -1]]).T
    print(B*B.T)
    print(B.T*B)

    print(U*U.T)
    print(U.T*U)
