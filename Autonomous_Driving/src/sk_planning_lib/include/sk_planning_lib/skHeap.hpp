#ifndef _SK_PLANNING_LIB_HEAP_H_
#define _SK_PLANNING_LIB_HEAP_H_

#include <vector>
#include <stdexcept>

#include "sk_planning_lib/skNode.hpp"

class skMinHeap
{
private:
    std::vector<skNode*> m_heap;
    unsigned int m_idx;

    void heapifyUp(int index)
    {
        int parentIndex = (index - 1) / 2;
        while( index > 0 && m_heap[index]->f(this->m_idx) < m_heap[parentIndex]->f(this->m_idx) )
        {
            std::swap(m_heap[index], m_heap[parentIndex]);
            index = parentIndex;
            parentIndex = (index - 1) / 2;
        }
    }

    void heapifyDown(int index)
    {
        int leftChildIndex = 2 * index + 1;
        int rightChildIndex = 2 * index + 2;
        int smallestIndex = index;

        if (leftChildIndex < m_heap.size() && m_heap[leftChildIndex]->f(this->m_idx) < m_heap[smallestIndex]->f(this->m_idx) )
        {
            smallestIndex = leftChildIndex;
        }
        if (rightChildIndex < m_heap.size() && m_heap[rightChildIndex]->f(this->m_idx) < m_heap[smallestIndex]->f(this->m_idx))
        {
            smallestIndex = rightChildIndex;
        }

        if (smallestIndex != index) {
            std::swap(m_heap[index], m_heap[smallestIndex]);
            heapifyDown(smallestIndex);
        }
    }

public:
    skMinHeap(unsigned int idx = 0) : m_idx(idx)
    {}

    void insert(skNode* value)
    {
        m_heap.push_back(value);
        heapifyUp(m_heap.size() - 1);
    }

    skNode* popMin()
    {
        if (m_heap.empty()) {
            throw std::out_of_range("Heap is empty");
        }

        skNode* min = m_heap.front();
        m_heap[0] = m_heap.back();
        m_heap.pop_back();
        heapifyDown(0);

        return min;
    }

    void updateHeap()
    {
        // 힙 전체를 재정렬
        for (int i = m_heap.size() / 2 - 1; i >= 0; --i) {
            heapifyDown(i);
        }
    }

    bool isEmpty() const
    {
        return m_heap.empty();
    }

    size_t size() const
    {
        return m_heap.size();
    }

    void clear()
    {
        this->m_heap.clear();
    }
};


#endif
