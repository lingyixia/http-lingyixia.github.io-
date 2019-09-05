---
title: DFSandBFS
date: 2019-06-23 10:20:44
category: 算法
tags: [DFS,BFS]
---
>>稍微总结一下DFS和BFS,一般情况下DFS用递归，BFS用队列

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

#迷宫最短路径问题
##dfs
```
#include <iostream>
#include <queue>

using namespace std;


vector<vector<int>> directions = {{0,  1},
                                  {0,  -1},
                                  {1,  0},
                                  {-1, 0}};
vector<vector<int>> finalPath;

void findShortPath_DFS(vector<vector<int>> &maze, vector<vector<int>> &currentPath, int startX, int startY, int endX, int endY)
{
    if (startX == endY && startY == endY)
    {
        finalPath = finalPath.size() < currentPath.size() && !finalPath.empty() ? finalPath : currentPath;
    }
    maze[startX][startY] = 1;
    for (auto direction:directions)
    {
        int currentX = startX + direction[0];
        int currentY = startY + direction[1];
        if (currentX >= 0 && currentX < maze.size() && currentY >= 0 && currentY < maze[0].size() &&
            maze[currentX][currentY] == 0)
        {
            currentPath.push_back(vector<int>{currentX, currentY});
            findShortPath_DFS(maze, currentPath, currentX, currentY, endX, endY);
            currentPath.pop_back();
            maze[currentX][currentY] = 0;
        }
    }
}

int main()
{
    vector<vector<int>> maze = {{0, 0, 1, 0, 0},
                                {0, 0, 0, 0, 0},
                                {0, 0, 0, 1, 0},
                                {1, 1, 0, 1, 1},
                                {0, 0, 0, 0, 0}};
    vector<vector<int>> currentPath;
    int startX = 0;
    int startY = 0;
    int endX = 4;
    int endY = 4;
    findShortPath_DFS(maze, currentPath, startX, startY, endX, endY);
    for (auto pos:finalPath)
    {
        cout << pos[0] << " " << pos[1] << endl;
    }
    return 0;
}
```

##bfs

```
#include <iostream>
#include <queue>

using namespace std;

struct Point
{
    int x;
    int y;

    Point() : x(0), y(0)
    {}

    Point(int x, int y) : x(x), y(y)
    {}
};

vector<vector<int>> directions = {{0,  1},
                                  {0,  -1},
                                  {1,  0},
                                  {-1, 0}};

void findShortPath_BFS(vector<vector<int>> &maze,
                       vector<vector<Point>> &pre,
                       int startX,
                       int startY,
                       int endX,
                       int endY)
{
    queue<Point> q;
    q.push(Point(startX, startY));
    maze[startX][startY] = 1;
    while (!q.empty())
    {
        Point current = q.front();
        q.pop();
        if (current.x == endX && current.y == endY)
        {
            break;
        }
        for (auto direction:directions)
        {
            int currentX = current.x + direction[0];
            int currentY = current.y + direction[1];
            if (currentX >= 0 && currentX < maze.size() && currentY >= 0 && currentY < maze[0].size() &&
                maze[currentX][currentY] == 0)
            {
                q.push(Point(currentX, currentY));
                maze[currentX][currentY] = 1;
                pre[currentX][currentY] = current;
            }
        }
    }
}

void print(Point point, vector<vector<Point>> &pre)
{
    if (point.x == 0 && point.y == 0)
    {
        cout << point.x << " " << point.y << endl;
        return;
    }
    print(pre[point.x][point.y], pre);
    cout << point.x << " " << point.y << endl;
}

int main()
{
    vector<vector<int>> maze = {{0, 0, 1, 0, 0},
                                {0, 0, 0, 0, 0},
                                {0, 0, 0, 1, 0},
                                {1, 1, 0, 1, 1},
                                {0, 0, 0, 0, 0}};
    vector<vector<Point>> pre(5, vector<Point>(5));
    int startX = 0;
    int startY = 0;
    int endX = 4;
    int endY = 4;
    findShortPath_BFS(maze, pre, startX, startY, endX, endY);
    print(Point(endX, endY), pre);
    return 0;
}
```