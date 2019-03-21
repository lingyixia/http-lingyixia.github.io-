---
title: 链表总结
date: 2019-02-02 11:43:08
categoty: 算法
tags: [链表]
---
>链表题型总结

#倒数第K个
>>给出一个单链表,输出该链表倒数第K个节点

三种解法:
##解法一
遍历一遍,记录长度L,然后重新遍历到第L-K的位置,代码略
##解法二
```
//递归,先递归到链表尾,然后往回走,每走一步K-1,如果K-1！=0则上抛NULL,反之就是找到了,上抛该节点即可
ListNode* FindKthToTail(ListNode* pListHead, unsigned int& k) //**注意引用传递的方式**
{
	if (!pListHead) return NULL;
	ListNode* node = FindKthToTail(pListHead->next, k);
	if (node)//找到了,不断上抛该节点
	{
		return node;
	}
	if (--k == 0)
	{
		return pListHead;//第一次找到,开始上抛
	}
	return NULL;
}
```

##解法三
```
//快慢指针1
ListNode* FindKthToTail(ListNode* pListHead, unsigned int k)
{
	int count = k;
	if (!pListHead || k == 0) return NULL;
	ListNode* behind = pListHead;
	ListNode* front = pListHead;
	bool flag = false;
	while (front)
	{
		if (count-- <= 0)
		{
			behind = behind->next;
			flag = true;
		}
		front = front->next;
	}
	return flag || count == 0 ? behind : NULL;
}
```

#判断有无环
一般解法有两种
##解法一
>>将遍历过的节点都加入哈希表,每次该点是否在哈希表中:

```
bool hasCycle(ListNode *head)
{
	unordered_set<ListNode*> record;
	ListNode* p = head;
	while (p)
	{
		if (record.count(p)>0)
		{
			return true;
		}
		record.insert(p);
		p = p->next;
	}
	return false;
}
```
##解法二
>>快慢指针,快指针一定会在环中与慢指针相遇,下面证明这个结论
命题:
如图所示:![](\img\cyclelinkllist.jpg)
注意,图中的$m,n$都是移动次数,即箭头个数,不是圆圈的个数!!!
其中快指针速度为$s$,慢指针速度为$q$且$s>q$,问为什么两者一定在环内相遇?
证明:假设慢指针第一次到达环内移动了$x$次,此时慢指针在环内的位置为$a$,则快指针也移动了$x$次,位置为$b$,则现在题目转化为:必有当继续走$y$步时使得$(a+s \times y)\mod n=(b+q \times y) \mod n$,即
$$
((b-a)+(q-s) \times y)\mod n =0 \tag{1}
$$
成立.
当移动$x$后有:
$$
a=(s \times x - m) \mod n \tag{2}
$$
$$
b=(q \times x -m) \mod n \tag{3}
$$
将$(2)(3)$带入$(1)$中:
$$
((q-s) \times x) \mod n +(q-s) \times y) \mod n =0
$$
由于:$((q-s) \mod n) \mod n =(q-s) \mod n$(多次取模等于一次取模)
故
$$
((q-s) \times (x+y)) \mod n=0 \tag{4}
$$
即只需要$(x+y)$是$n$的整数倍即可.因为我们一般取$q=2,s=1$,因为当$q=2,s=1$的时候最快取得$n$得整数倍,即循环次数最少。

```
bool hasCycle1(ListNode *head)
{
	ListNode* slow = head;
	ListNode* fast = head;
	while (fast)
	{
		fast = fast->next;
		if (!fast) return false;
		fast = fast->next;
		slow = slow->next;
		if (fast == slow)
		{
			return true;
		}
	}
	return false;
}
```
#判断环起始位置
>>求起始位置也就是求上图中$m$得长度,由于$p=2,s=1,x=m$,相遇位置为$y$,$y$是距离环起始位置的长度,上图(4)式可改为:
$$
(m+y) \mod n =0
$$
但是在这里要写成这样的形式:
$$
(y+m) \mod n =0 \tag{5}
$$
即$m$是在$y$的基础上再增加,公式(5)和公(4)意义不同,公式(4)指的是当符合公式(4)的时候前后两指针相遇,公式(5)的意义是当符合该公式的时候对$m$取模为零,即正好在环头位置.
eg:
![](\img\imgcyclelinkllist2.png)
图中在节点5处相遇,此时$m+y$就是$1,2,3,4,5$,此时有$(m+y) \mod n=0$,现在把这个长度的前$m$放到后面,$3,4,5,6,3$,即(y+y),此时同样有$(y+m)\mod n =0$,因此现在要做的就是一个指针放到头部,一个指针放到相遇的地方,依次next一步,当两者相遇的时候就是环的初始位置.

```
ListNode *detectCycle(ListNode *head) {
        ListNode* slow=head;
        ListNode* fast = head;
        while(fast)
        {
            fast=fast->next;
            if(!fast) return NULL;
            fast=fast->next;
            slow=slow->next;
            if(fast==slow)
            {
                break;
            }
        }
        if(!fast) return NULL;
        ListNode* p =head;
        ListNode* q=fast;
        while(p!=q)
        {
            p=p->next;
            q=q->next;
        }
        return p;
    }
```
#删除链表重复元素
解法一:
```
struct ListNode
{
	int val;
	ListNode* next;
	ListNode(int x) :val(x), next(NULL) {}
};
void deleteOne(ListNode* head,ListNode* current)//注意删除单节点方式
{
	if (current->next)
	{
		current->val = current->next->val;
		current->next = current->next->next;
	}
	else
	{
		ListNode* p = head;//如果删除的是尾节点则需要从头遍历
		while (p->next!=current)
		{
			p = p->next;
		}
		p->next = NULL;
	}
}
ListNode* deleteDuplicates(ListNode* pHead)
{
	ListNode* newHead = new ListNode(-100);
	newHead->next = pHead;
	ListNode* before = newHead;
	ListNode* after = pHead;
	bool flag = false;
	while (after)
	{
		while (after &&( before->val == after->val))
		{
			flag = true;
			deleteOne(newHead,after);
			after = before->next;
		}
		if (flag)
		{
			deleteOne(newHead,before);
			flag = false;
		}
		else
		{
			before = after;
		}
		after = before->next;
	}
	return newHead->next;
}
```

解法二:
```
 ListNode* deleteDuplication(ListNode* pHead)
    {
       if(!pHead || !pHead->next) return pHead;
       if(pHead->next->val == pHead->val)
       {
           while(pHead->next && pHead->next->val == pHead->val)
           {
               pHead=pHead->next;
           }
           return deleteDuplication(pHead->next);
       }
       pHead->next = deleteDuplication(pHead->next);
       return pHead;
    }
```