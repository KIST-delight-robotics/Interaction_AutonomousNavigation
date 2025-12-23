#ifndef _SK_PLANNING_LIB_GRAPH_H_
#define _SK_PLANNING_LIB_GRAPH_H_

#include <vector>
#include <stdexcept>

#include "sk_planning_lib/skMapCell.h"
#include "sk_planning_lib/skNode.hpp"

using namespace std;

class skGraph
{
protected:
    skState m_goal;
    skState m_start;

public:
    skGraph()
    {}

    ~skGraph()
    {}

    void setGoal(const skState goal)
    {
        this->m_goal = goal;
    }

    bool isGoal(const skState node) const
    {
        return (this->m_goal == node);
    }

    std::vector<skNode> getSuccessors(const skNode& p)
    {
        std::vector<skNode> c;
        c.clear();

        skNode n;
        double cost;
        for(unsigned int j = 0; j < p.m_state.getMaxNeighbor(); j++ )
        {
            n = p;
            cost = n.m_state.moveTo(j);            
            if( cost < 0.0 )
            {
                continue;;
            }
            if( n.m_state.isFeasible() )
            {
                n.m_g += cost;
                n.m_h = n.m_state.getHeuristic(this->m_goal);
                c.push_back(n);
            }
        }

        return (c);
    }

    friend class skAStar;
};


#endif
