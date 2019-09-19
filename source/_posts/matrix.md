---
title: 矩阵若干题解
date: 2019-05-07 16:47:53
category: 算法
tags: [动态规划,回溯,dfs]
---

>矩阵的题往往和回溯有关,然后在回溯的基础上用动态规划提升效率

#机器人路径
>>地上有一个m行和n列的方格。一个机器人从坐标0,0的格子开始移动，每一次只能向左，右，上，下四个方向移动一格，但是不能进入行坐标和列坐标的数位之和大于k的格子。 例如，当k为18时，机器人能够进入方格（35,37），因为3+5+3+7 = 18。但是，它不能进入方格（35,38），因为3+5+3+8 = 19。请问该机器人能够达到多少个格子？

```
bool checkout(int threshold, int row, int col, vector <vector<bool>> &visited)
{
    if (row >= 0 && row < visited.size() && col >= 0 && col < visited[0].size() && !visited[row][col] &&
        getSum(row) + getSum(col) <= threshold)
        return true;
    return false;
}

int getSum(int data)
{
    int sum = 0;
    while (data != 0)
    {
        sum += data % 10;
        data = data / 10;
    }
    return sum;
}

int dfs(vector <vector<bool>> &visited, int threshold, int row, int col)
{
    int count = 0;
    if (checkout(threshold, row, col, visited))
    {
        visited[row][col] = true;
        count = 1 +
                dfs(visited, threshold, row + 1, col) +
                dfs(visited, threshold, row - 1, col) +
                dfs(visited, threshold, row, col + 1) +
                dfs(visited, threshold, row, col - 1);
    }
    return count;
}

int movingCount(int threshold, int rows, int cols)
{
    vector <vector<bool>> visited;
    for (int i = 0; i < rows; i++)
    {
        vector<bool> temp(cols, false);
        visited.push_back(temp);
    }
    int count = moveCountCore(visited, threshold, 0, 0);
    return count;
}
```

#矩阵中的路径
>>请设计一个函数，用来判断在一个矩阵中是否存在一条包含某字符串所有字符的路径。路径可以从矩阵中的任意一个格子开始，每一步可以在矩阵中向左，向右，向上，向下移动一个格子。如果一条路径经过了矩阵中的某一个格子，则之后不能再次进入这个格子。 例如 a b c e s f c s a d e e 这样的3 X 4 矩阵中包含一条字符串"bcced"的路径，但是矩阵中不包含"abcb"路径，因为字符串的第一个字符b占据了矩阵中的第一行第二个格子之后，路径不能再次进入该格子。

```
bool dfs(char *matrix, int rows, int cols, int row, int col, vector <vector<bool>> &visited, int indexLength, char *str)
{
    if (str[indexLength] == '\0')
    {
        return true;
    }
    bool has = false;
    if (row >= 0 && row < rows && col >= 0 && col < cols && !visited[row][col] &&
        matrix[row * cols + col] == str[indexLength])
    {
        visited[row][col] = true;
        has = dfs(matrix, rows, cols, row + 1, col, visited, indexLength + 1, str) ||
              dfs(matrix, rows, cols, row - 1, col, visited, indexLength + 1, str) ||
              dfs(matrix, rows, cols, row, col + 1, visited, indexLength + 1, str) ||
              dfs(matrix, rows, cols, row, col - 1, visited, indexLength + 1, str);
        if (!has)
        {
            visited[row][col] = false;
        }
    }
    return has;
}

bool hasPath(char *matrix, int rows, int cols, char *str)
{
    vector <vector<bool>> visited;
    for (int i = 0; i < rows; i++)
    {
        vector<bool> temp(cols, false);
        visited.push_back(temp);
    }
    for (int i = 0; i < rows; i++)
    {
        for (int j = 0; j < cols; j++)
        {
            if (findPathCore(matrix, rows, cols, i, j, visited, 0, str))
            {
                return true;
            }
        }
    }
    return false;
}
```

#矩阵中矩阵最长递增路径
>>[一维最长递增子序列](https://lingyixia.github.io/2019/04/11/DynamicProgramming/#%E6%9C%80%E9%95%BF%E9%80%92%E5%A2%9E%E5%AD%90%E5%BA%8F%E5%88%97)
```
vector<vector<int>> directions = { {0,1},{0,-1},{1,0},{-1,0} };
int dfs(vector<vector<int>>& matrix,vector<vector<int>>& dp, int posX, int posY)
{
	if (dp[posX][posY]) return dp[posX][posY];
	int maxLength = 0;
	for (auto direction : directions)
	{
		int currentX = posX + direction[0];
		int currentY = posY + direction[1];
		if (currentX >= 0 && currentX < matrix.size() && currentY >= 0 && currentY < matrix[0].size() && matrix[currentX][currentY] > matrix[posX][posY])
		{
			maxLength = max(maxLength, dfs(matrix,dp, currentX, currentY));
		}
	}
	dp[posX][posY] = maxLength+1;
	return dp[posX][posY];
}

int longestIncreasingPath(vector<vector<int>>& matrix) {
	if (matrix.empty() || matrix[0].empty()) return 0;
	int maxLength = 0;
	vector<vector<int>> dp(matrix.size(),vector<int>(matrix[0].size()));
	for (int i = 0; i < matrix.size(); i++)
	{
		for (int j = 0; j < matrix[0].size(); j++)
		{
			maxLength = max(maxLength, dfs(matrix,dp, i, j));
		}
	}
	return maxLength;
}
```
#[八皇后](https://leetcode.com/problems/n-queens)
```
class Solution {
public:
    vector<vector<string>> result;
bool judge(vector<int> current,int column)
{
	for (int row = 0; row < current.size(); row++)
	{
		if (current[row]==column || abs(column-current[row])==abs(int(current.size())-row))
		{
			return false;
		}
	}
	return true;
}
void queens(vector<int> current,int n)
{
	if (current.size()==n)
	{
		vector<string> temp;
		for (int row = 0; row < current.size(); row++)
		{
			string str = "";
			for (int col = 0; col < current.size(); col++)
			{
				if (col==current[row])
				{
					str += "Q";
				}
				else
				{
					str += ".";
				}
			}
			temp.push_back(str);
		}
		result.push_back(temp);
	}
	else
	{
		for (int col = 0; col < n; col++)
		{
			if (judge(current,col))
			{
				current.push_back(col);
				queens(current,n);
				current.pop_back();
			}
		}
	}
}
vector<vector<string>> solveNQueens(int n)
{
	vector<int> current;
	queens(current,n);
	return result;
}
};
```