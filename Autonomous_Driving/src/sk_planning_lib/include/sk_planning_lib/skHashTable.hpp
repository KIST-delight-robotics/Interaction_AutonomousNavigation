#ifndef _SK_PLANNING_LIB_HASHTABLE_H_
#define _SK_PLANNING_LIB_HASHTABLE_H_

#include "rclcpp/rclcpp.hpp"
#include <vector>
#include <stdlib.h>
#include <stdio.h>
#include <unordered_set>

#include <chrono>
#include <functional>
#include <memory>
#include <string>

#include "sk_planning_lib/skNode.hpp"

using namespace std;

class MyHashFunction
{
public:
    size_t operator()(const skNode& node) const
    {
        return (node.hash());
    }
};

class skNodeHashtable
{
private:
    std::unordered_set<skNode, MyHashFunction> m_nodes;

public:
    skNodeHashtable()
    {
        this->m_nodes.clear();
    }

    ~skNodeHashtable()
    {
        for (const skNode& node : this->m_nodes) {
            delete &node;
        }
        this->m_nodes.clear();
    }

    bool addNode(skNode* node)
    {
        if (this->m_nodes.find(*node) != this->m_nodes.end())
        {
            return (false);
        }
        else
        {
            this->m_nodes.insert(*node);
            return (true);
        }
    }

    skNode* getNode(const skNode& targetNode)
    {
        std::unordered_set<skNode>::iterator it = this->m_nodes.find(targetNode);
        if (it != this->m_nodes.end())
        {
            return const_cast<skNode*>(&(*it));
        }
        else
        {
            return (nullptr);
        }
    }

    void removeNode(const skNode& targetNode)
    {
        std::unordered_set<skNode>::iterator it = this->m_nodes.find(targetNode);
        if (it != this->m_nodes.end())
        {
            delete (&(*it));
            this->m_nodes.erase(it);
        }
    }

    bool isEmpty() const
    {
        return (this->m_nodes.empty());
    }

    size_t size() const
    {
        return (this->m_nodes.size());
    }

    void clear()
    {
        this->m_nodes.clear();
    }
};

#endif
