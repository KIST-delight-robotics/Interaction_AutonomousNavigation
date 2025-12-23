#ifndef _SK_PLANNING_LIB_NODE_H_
#define _SK_PLANNING_LIB_NODE_H_

#include <vector>
#include <stdlib.h>
#include <stdio.h>

#include <chrono>
#include <functional>
#include <memory>
#include <string>

#include "sk_planning_lib/skGrid2D.h"

using namespace std;

class skNode
{
protected:
    static unsigned int s_number_of_heuristics;

public:
    void setNumberOfHeuristics(const int& n)
    {
        this->s_number_of_heuristics = n;
    }

    unsigned int getNumberOfHeuristics() const
    {
        return (this->s_number_of_heuristics);
    }

protected:
    skState m_state;
    //std::vector<double> m_f;
    double m_g;
    std::vector<double> m_h;

    skNode* p_parrent;

public:
    skNode() : p_parrent(NULL)
    {
        this->m_h.resize(this->s_number_of_heuristics,0.0);
    }

    ~skNode()
    {}

    double f(const unsigned int& idx = 0) const
    {
        //return (this->m_f[idx]);
        return (this->m_g+this->m_h[idx]);
    }

    double g() const
    {
        return (this->m_g);
    }

    double h(const unsigned int& idx = 0) const
    {
        return (this->m_h[idx]);
    }

    size_t hash() const
    {
        return (this->m_state.hash());
    }
 
    friend class skGraph;
    friend class skAStar;

    friend bool operator==(const skNode& lhs, const skNode& rhs);
    friend bool operator!=(const skNode& lhs, const skNode& rhs);
};

bool operator==(const skNode& lhs, const skNode& rhs)
{
    return (lhs.m_state==rhs.m_state);
}

bool operator!=(const skNode& lhs, const skNode& rhs)
{
    return (lhs.m_state!=rhs.m_state);
}

#endif
