# -*- coding: UTF-8 -*-

import numpy as np
from numpy import linalg as La


class MatrixFactorization:
    matA = ''  # 读入的矩阵
    matA_backup = ''  # A矩阵的备份
    op = ''  # 操作类型
    result = {}  # 运行结果示例.txt,不同的分解结果是不同的键值对

    def readMat(self, path):
        with open(path) as f:
            try:
                lines = f.readlines()
                nums = [[int(rc.strip('\n')) for rc in c.split(',')] for c in lines]

                # 不使用科学计数法
                np.set_printoptions(suppress=True)
                self.matA = np.array(nums, dtype=np.float)
                self.matA_backup = self.matA.copy()
            except:
                print("File Not Found/Data Format Exception!")
                exit(-1)

    def readOp(self, op):
        self.op = op = op.upper()
        if op == "LU":
            self.result = self.LU(self.matA)
        elif op == "GS":
            self.result = self.GramSchmidt(self.matA)
        elif op == "HH":
            self.result = self.HouseHolder(self.matA)
        elif op == "GI":
            self.result = self.Givens(self.matA)
        else:
            print("Undefined Operation Method!")

    def SwapRow(self, mat, ra, rb):
        res = mat
        res[[ra, rb], :] = mat[[rb, ra], :]
        return res

    def maxPivotIndex(self, matCol, rowStart):
        piv = rowStart
        for i in range(rowStart, matCol.__len__()):
            if abs(matCol[i]) > abs(matCol[piv]):
                piv = i
        return piv

    def LU_Gaussian(self, mat, L, piv):
        """LU高斯消元,第i行i列,在这之前mat[i][i]已经是最大"""
        row, col = mat.shape
        for i in range(piv + 1, row):
            # 获得消元的参数,放入L
            L[i][piv] = mat[i][piv] / mat[piv][piv]
            for j in range(piv, col):
                # 对这一行的列进行消元操作
                mat[i][j] = mat[i][j] - L[i][piv] * mat[piv][j]
        return L, mat

    def LU(self, mat):
        """LU分解"""
        row, col = mat.shape
        # P = np.arange(row) + 1 # 增广列P
        matP = np.eye(row, col)  # P矩阵对应的矩阵
        L = np.zeros(mat.shape)  # 下三角矩阵L
        for i in range(row):
            piv = self.maxPivotIndex(mat[:, i], i)
            mat = self.SwapRow(mat, i, piv)
            matP = self.SwapRow(matP, i, piv)
            L = self.SwapRow(L, i, piv)
            L, mat = self.LU_Gaussian(mat, L, i)

        return {
            "L": L,
            "U": mat,
            "P": matP
        }

    def GramSchmidt(self, mat):
        """施密特正交化"""
        row, col = mat.shape
        Q = mat
        R = np.zeros(mat.shape)
        for j in range(col):
            for i in range(j):
                # Rij = qT.x, qj =
                R[i, j] = np.dot(Q[:, i].T, mat[:, j])
                Q[:, j] = Q[:, j] - R[i, j] * Q[:, i]
            # qi = qi/Rii
            R[j, j] = La.norm(Q[:, j])
            Q[:, j] = Q[:, j] / R[j, j]

        return {
            "Q": Q,
            "R": R
        }

    def HouseHolder(self, mat):
        """HouseHolder约减"""
        row, col = mat.shape
        P = np.eye(row)
        for i in range(col):
            matA = mat[i:, i:]
            matAc = matA[:, 0]
            matE = np.eye(matAc.shape[0])
            matU = matAc - La.norm(matAc) * matE[:, 0]
            matU = matU.reshape((matAc.shape[0], 1))

            if np.dot(matU.T, matU) != 0:
                matR = matE - 2 * np.dot(matU, matU.T) / np.dot(matU.T, matU)
            else:
                matR = matE

            matA = np.dot(matR, matA)
            R = np.eye(row)
            # R2 需要增广一维单位阵
            R[i:, i:] = np.copy(matR)
            P = np.dot(R, P)
            mat[i:, i:] = matA

        # PA = T
        return {
            "Q": P.T,
            "R": mat
        }

    def RotateMat(self, mat, i, j):
        """计算旋转矩阵"""
        row, col = mat.shape
        Q = np.eye(row)
        cVal = sum(c**2 for c in mat[j:i, j])
        mVal = (cVal + mat[i][j]**2) ** 0.5
        c = cVal**0.5 / mVal
        s = mat[i][j] / mVal
        Q[i][i], Q[j][j] = c, c
        Q[i][j], Q[j][i] = -s, s
        return Q

    def Givens(self, mat):
        """Givens约减"""
        row, col = mat.shape
        Q = np.eye(row)
        for j in range(col):
            # 对主元下方消成0
            for i in range(j+1, row):
                rotMat = self.RotateMat(mat, i, j)
                Q = np.dot(rotMat, Q)
                mat = np.dot(rotMat, mat)

        return {
            "Q": Q.T,
            "R": mat
        }

    def printResult(self):
        """输出结果"""
        print("原矩阵:")
        print(self.matA_backup)
        print("分解后的矩阵:")
        for (k, v) in self.result.items():
            print(k, ": ")
            print(v)

if __name__ == "__main__":
    obj = MatrixFactorization()
    print("========================Start===============================")
    print("矩阵分解算法:Matrix Factorization Algorithm(LU,Gram-Schmidt,HouseHolder,Givens)")
    print("Input file Path:")
    obj.readMat(input())
    # obj.readMat("data/matrixQR.txt")
    obj.readOp(input("操作类型(LU,GS,HH,GI):"))

    print("================Results==================")
    obj.printResult()

