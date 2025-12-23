#ifndef _SK_PLANNING_LIB_GRID_2D_H_
#define _SK_PLANNING_LIB_GRID_2D_H_

//#include <stdio.h>
#include <iostream>
#include <fstream>
#include <vector>

#include "sk_planning_lib/skState.hpp"
//#include "sk_planning_lib/skMap2D.hpp"

using namespace std;

class skGrid2D : public skState
{
protected:
    int m_x;
    int m_y;

public:
    
    skGrid2D() : m_x(0), m_y(0)
    {}

    skGrid2D(const int& _x, const int& _y) : m_x(_x), m_y(_y)
    {}

    ~skGrid2D()
    {}

    double moveTo(const int& j)
    {
        if( j == 0 )
        {
            this->m_x--;
            return (1.0);
        }
        else if( j == 1 )
        {
            this->m_x++;
            return (1.0);
        }
        else if( j == 2 )
        {
            this->m_y--;
            return (1.0);
        }
        else if( j == 3 )
        {
            this->m_y++;
            return (1.0);
        }
        else if( j == 4 )
        {
            this->m_x--;
            this->m_y--;
            return (sqrt(2.0));
        }
        else if( j == 5 )
        {
            this->m_x--;
            this->m_y++;
            return (sqrt(2.0));
        }
        else if( j == 6 )
        {
            this->m_x++;
            this->m_y--;
            return (sqrt(2.0));
        }
        else if( j == 7 )
        {
            this->m_x++;
            this->m_y++;
            return (sqrt(2.0));
        }

        return (-10.0);
    }

    size_t hash() const
    {
        return (this->m_x);
    }

    bool operator==(const skGrid2D& rhs) const
    {
        return ((this->m_x == rhs.m_x)&&(this->m_y == rhs.m_y));
    }

    bool operator!=(const skGrid2D& rhs) const
    {
        return (!(*this==rhs));
    }

    friend class skMapCell;
    friend class skMapDist;
    friend class skGraph;
};

#endif // _SK_PLANNING_LIB_GRID_2D_H_
