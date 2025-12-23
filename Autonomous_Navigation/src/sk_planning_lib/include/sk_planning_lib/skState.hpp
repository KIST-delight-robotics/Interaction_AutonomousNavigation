#ifndef _SK_PLANNING_LIB_STATE_H_
#define _SK_PLANNING_LIB_STATE_H_

#include <vector>
#include <stdlib.h>
#include <stdio.h>

#include <chrono>
#include <functional>
#include <memory>
#include <string>

using namespace std;

/*=======================================================================
    Reference Class to describe the state of agent or robot
    Need to inherit this class to make a new state
    1. Make proper varibles to describe state
    2. Make method to find proper successors for each idx < s_number_of_edges
    3. Do not forget to assign proper cost for each case
    4. Make method of isFeasible
=======================================================================*/
//class skGrpah;

class skState
{
protected:
    static unsigned int s_max_neighbor;

public:
    skState(){}
    ~skState(){}

    void setMaxNeighbor(const unsigned int& n)
    {
        this->s_max_neighbor = n;
    }

    unsigned int getMaxNeighbor() const
    {
        return (this->s_max_neighbor);
    }

    virtual double moveTo(const unsigned int& idx)
    {
        return (-10.0);
    }

    virtual bool isFeasible() const
    {
        return (false);
    }

    virtual std::vector<double> getHeuristic(const skState& n) const
    {
        std::vector<double> h;
        h.resize(1,0.0);

        return (h);
    }

    virtual bool operator==(const skState& rhs) const
    {
        return (true);
    }

    virtual bool operator!=(const skState& rhs) const
    {
        return (false);
    }

    virtual size_t hash() const
    {
        return (0);
    }

    friend class skGraph;
};

#endif
