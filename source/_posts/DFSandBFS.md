---
title: DFSandBFS
date: 2019-06-23 10:20:44
category: 算法
tags: [DFS,BFS]
---
>>稍微总结一下DFS和BFS

#有向图的遍历
##DFS
```
#include<iostream>
#include<vector>
#include<queue>

using namespace std;

void dfs(vector<bool> &visisted, vector<vector<int>> &neibourhoods, int index)
{
    visisted[index] = true;
    cout << index << "-->";
    for (int i = 0; i < neibourhoods[index].size(); ++i)
    {
        if (!visisted[neibourhoods[index][i]])
        {
            dfs(visisted, neibourhoods, neibourhoods[index][i]);
        }
    }
}


void dfsInit(int numCourses, vector<vector<int>> &neibourhoods)
{
    vector<bool> visisted(numCourses, false);
    for (int i = 0; i < numCourses; ++i)
    {
        if (!visisted[i])
        {
            dfs(visisted, neibourhoods, i);
        }
    }
}

int main()
{
    vector<vector<int>> neibourhoods = {{1, 3},{2, 3},{},{4},{5, 6},{2},{3}};
    dfsInit(7, neibourhoods);
    return 0;
}
```

##BFS
>>其实就是层序遍历 
```
#include<iostream>
#include<vector>
#include<queue>

using namespace std;

void bfs(int numCourses, vector<vector<int>> &neibourhoods)
{
    vector<bool> visisted(numCourses, false);
    queue<int> q;
    q.push(0);//其实就是树的层序遍历，首先需要知道一个入度为零的点
    visisted[0] = true;
    while (!q.empty())
    {
        int current = q.front();
        cout << current << "-->";
        q.pop();
        for (auto node:neibourhoods[current])
        {
            if (!visisted[node])
            {
                q.push(node);
                visisted[node] = true;
            }
        }
    }
}

int main()
{
    vector<vector<int>> neibourhoods = {{1, 3},{2, 3},{},{4},{5, 6},{2},{3}};
    bfs(7, neibourhoods);
    return 0;
}
```