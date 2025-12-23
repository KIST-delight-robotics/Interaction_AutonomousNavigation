#include "sk_planning_lib/skGrid2D.h"
#include "sk_planning_lib/skMapCell.h"

using namespace std;

skMapCell* skGrid2D::p_map = NULL;

void skGrid2D::setMap(skMapCell* map)
{
    p_map = map;
}

std::vector<double> skGrid2D::getHeuristic(const skGrid2D& n) const
{
    std::vector<double> h;
    h.resize(1,0.0);

    return (h);
}

double skGrid2D::moveTo(const unsigned int& j)
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

bool skGrid2D::isFeasible() const
{
    return (this->p_map->isFeasible(this->m_x, this->m_y));
}

size_t skGrid2D::hash() const
{
    return (this->m_x);
}

bool skGrid2D::operator==(const skGrid2D& rhs) const
{
    return ((this->m_x == rhs.m_x)&&(this->m_y == rhs.m_y));
}

bool skGrid2D::operator!=(const skGrid2D& rhs) const
{
    return (!(*this==rhs));
}

int skGrid2D::x() const
{
    return (this->m_x);
}

int skGrid2D::y() const
{
    return (this->m_y);
}

double skGrid2D::distETo(const skGrid2D& rhs) const
{
    const int dsq((this->m_x-rhs.m_x)*(this->m_x-rhs.m_x)+(this->m_y-rhs.m_y)*(this->m_y-rhs.m_y));

    const double dist(sqrt((double)dsq));

    if( this->p_map )
        return (this->p_map->m_grid_size*dist);

    return (dist);
}

double skGrid2D::dist8To(const skGrid2D& rhs) const
{
    const int dx(abs(this->m_x-rhs.m_x));
    const int dy(abs(this->m_y-rhs.m_y));

    const double dist(M_SQRT2*(double)(MIN(dx,dy))+(double)(abs(dx-dy)));

    if( this->p_map )
        return (this->p_map->m_grid_size*dist);

    return (dist);
}
