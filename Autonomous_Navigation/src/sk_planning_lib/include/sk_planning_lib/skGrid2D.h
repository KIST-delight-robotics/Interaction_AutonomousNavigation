#ifndef _SK_PLANNING_LIB_GRID_2D_H_
#define _SK_PLANNING_LIB_GRID_2D_H_

//#include <stdio.h>
#include <iostream>
#include <fstream>
#include <vector>

#include "sk_planning_lib/skState.hpp"
#include "sk_planning_lib/skGrid2D.h"
#include "sk_planning_lib/skMapCell.h"

using namespace std;

class skMapCell;

class skGrid2D : public skState
{
protected:
    int m_x;
    int m_y;

    static skMapCell* p_map;

public:
    
    skGrid2D() : m_x(0), m_y(0)
    {}

    skGrid2D(const int& _x, const int& _y) : m_x(_x), m_y(_y)
    {}

    ~skGrid2D()
    {}

    static void setMap(skMapCell* map);

    std::vector<double> getHeuristic(const skGrid2D& n) const;
    double moveTo(const unsigned int& j);
    bool isFeasible() const;
    size_t hash() const;
    bool operator==(const skGrid2D& rhs) const;
    bool operator!=(const skGrid2D& rhs) const;
    int x() const;
    int y() const;
    double distETo(const skGrid2D& rhs) const;
    double dist8To(const skGrid2D& rhs) const;

    friend class skMapCell;
    friend class skMapDist;
    friend class skGraph;
};

#endif // _SK_PLANNING_LIB_GRID_2D_H_
