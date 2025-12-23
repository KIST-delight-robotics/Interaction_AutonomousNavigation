#ifndef _SK_PLANNING_LIB_ASTAR_H_
#define _SK_PLANNING_LIB_ASTAR_H_

#include <vector>
#include <stdexcept>

#include "sk_planning_lib/skGraph.hpp"
#include "sk_planning_lib/skHashTable.hpp"
#include "sk_planning_lib/skHeap.hpp"

using namespace std;

class skAStar
{
protected:
    skGraph m_graph;
    skNodeHashtable m_hash;
    skMinHeap m_heap;
    skNode* p_path;

public:
    skAStar()
    {}

    ~skAStar()
    {}

    // Setting grpah and problems
    void setMap(skMapCell* map)
    {
        skGrid2D::setMap(map);
    }

    bool initProblem(const skGrid2D& start, const skGrid2D& goal)
    {
        // Check and set goal
        if( !goal.isFeasible() )
            return (false);
        this->m_graph.setGoal(goal);

        // Check and set start
        if( !start.isFeasible() )
            return (false);
        skNode init;
        init.setNumberOfHeuristics(1);
        init.m_state = start;
        init.m_g = 0.0;
        init.m_h = start.getHeuristic(goal);
        if( this->m_hash.addNode(&init) )
        {
            skNode* p_init = this->m_hash.getNode(init);
            this->m_heap.insert(p_init);
        }

        this->p_path = NULL;

        return (true);
    }

    void clear()
    {
        this->m_hash.clear();
        this->m_heap.clear();
    }

    // Run A*
    bool find_path()
    {
        skNode* p_node;
        while( !this->m_heap.isEmpty() )
        {
            // Find node to expand
            p_node = this->m_heap.popMin();

            // check if it is goal
            /*if( this->m_graph.isGoal((skGrid2D)(p_node->m_state)) )
            {
                this->p_path = p_node;
                return (true);
            }*/

            // expand
            this->expand(p_node);
        }

        return (false);
    }

    // Expand Node
    void expand(skNode* p)
    {
        std::vector<skNode> c = this->m_graph.getSuccessors(*p);
/*        for( unsigned int j = 0; j < c.size(); j++ )
        {
            if( this->m_graph.isFeasible(c[j].m_node) )
            {
                if( this->m_hash.addNode(*(c[j])) )
                {
                    this->m_heap
                }
            }
        }
*/
    }

    // Add node
    void insert(skNode n)
    {
        
    }
};


#endif
