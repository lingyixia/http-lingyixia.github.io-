---
title: convex
date: 2018-12-27 20:20:55
category: 算法
tags: [C++,leetcode]
description: 对应[leetcode 587](https://leetcode.com/problems/erect-the-fence/)
---

>There are some trees, where each tree is represented by (x,y) coordinate in a two-dimensional garden. Your job is to fence the entire garden using the minimum length of rope as it is expensive. The garden is well fenced only if all the trees are enclosed. Your task is to help find the coordinates of trees which are exactly located on the fence perimeter.

# Graham扫描法
步骤:
1. 将所有的点排序,取$y$轴最小的一个点记为$P_0$(如果多个则取$x$最小的点)
2. 平移所有的点,使$P_0$为原点
3. 计算各个点相对于$P_0$的夹角$α$,按从小到大的顺序对各个点排序。当$α$相同时，距离$P_0$比较近的排在前面。例如图中得到的结果为$P_1,P_2,P_3,P_4,P_5,P_6,P_7,P_8$(**此处排序是个关键点**)。我们由几何知识可以知道，结果中第一个点$P_1$和最后一个点$P_8$一定是凸包上的点。然后将$P_0$和$P_1$依次入栈。
4. 栈顶出栈,即$P_1$然后判断下一个点$P_2$在栈顶点$P_0$和刚刚出栈的点$P_1$组成的直线的哪边，如果在左边或者同一直线上则$P_1$再次入栈,$P_2$跟随入栈,如果在右边则$P_1$扔掉,扔掉，重复步骤4,直到$P_2$在该直线的左边或同一直线[**此处判断是个关键点**](/2018/12/27/convex/)。
5. 直到最后一个$P_8$为止:
代码如下(需要简化,以后在说):
```
#include<iostream>
#include<algorithm>
#include<stack>
#include<vector>
using namespace std;
struct  Point
{
	int x, y;
	Point() :x(0), y(0) {}
	Point(int a, int b) :x(a), y(b) {}
};
Point operator -(const Point& pointA, const Point& pointB)
{
	Point temp;
	temp.x = pointA.x - pointB.x;
	temp.y = pointA.y - pointB.y;
	return temp;
}
Point operator +(const Point& pointA, const Point& pointB)
{
	Point temp;
	temp.x = pointA.x + pointB.x;
	temp.y = pointA.y + pointB.y;
	return temp;
}
bool compareLess(const Point& A, const Point& B)
{
	if (A.y == B.y)
	{
		return A.x < B.x;
	}
	return A.y < B.y;
}
bool compareAngle(const Point& A, const Point& B)
{
	float a = A.x / sqrt(A.x*A.x + A.y* A.y);
	float b = B.x / sqrt(B.x*B.x + B.y * B.y);
	if (a == b)
	{
		if (A.x == B.x)
		{
			return A.y > B.y;
		}
		return A.x < B.x;
	}
	return a > b;
}
vector<Point> outerTrees(vector<Point>& points);
int calcSquare(Point& A, Point& B, Point& C);
int main()
{
	vector<Point> points = { Point(0, 2), Point(0, 4), Point(0, 5), Point(0, 9), Point(2, 1), Point(2, 2), Point(2, 3), Point(2, 5), Point(3, 1), Point(3, 2), Point(3, 6), Point(3, 9), Point(4, 2), Point(4, 5), Point(5, 8), Point(5, 9), Point(6, 3), Point(7, 9), Point(8, 1), Point(8, 2), Point(8, 5), Point(8, 7), Point(9, 0), Point(9, 1), Point(9, 6) };
	vector<Point> p = outerTrees(points);
	return 0;
}
vector<Point> outerTrees(vector<Point>& points)
{
	if (points.size() < 4)
	{
		return points;
	}
	sort(points.begin(), points.end(), compareLess);
	Point zero = points.front();
	for (vector<Point>::iterator iter = points.begin(); iter != points.end(); iter++)
	{
		*iter = *iter - zero;
	}
	sort(points.begin() + 1, points.end(), compareAngle);
	vector<Point>::iterator iter;
	for (iter = points.begin(); iter->x == 0; iter++);
	sort(points.begin(), iter, compareLess);
	stack<Point> outer;
	vector<Point> result;
	outer.push(points.front());
	outer.push(*(points.begin() + 1));
	for (vector<Point>::iterator iter = points.begin() + 2; iter != points.end(); iter++)
	{
		Point tempTop = outer.top();
		outer.pop();
		while (calcSquare(outer.top(), tempTop, *iter) < 0)
		{

			tempTop = outer.top();
			outer.pop();
		}
		outer.push(tempTop);
		outer.push(*iter);
	}
	while (!outer.empty())
	{
		result.push_back(outer.top() + zero);
		outer.pop();
	}
	return result;
}
int calcSquare(Point& A, Point& B, Point& C)
{
	return A.x * B.y + A.y*C.x + B.x*C.y - B.y*C.x - A.y*B.x - A.x*C.y;
}
```