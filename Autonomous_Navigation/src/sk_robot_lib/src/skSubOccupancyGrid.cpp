#include "sk_robot_lib/skSubOccupancyGrid.h"

skSubOccupancyGrid::skSubOccupancyGrid() : m_distMap(false)
{
}

skSubOccupancyGrid::~skSubOccupancyGrid()
{
}

bool skSubOccupancyGrid::buildDistMap(const double& maxDist)
{
    if( !this->haveMsg() )
    {
        this->m_distMap = false;
        return (false);
    }
    
    this->m_resolution = this->m_msg.info.resolution;
    this->m_width = this->m_msg.info.width;
    this->m_height = this->m_msg.info.height;
    this->m_x = this->m_msg.info.origin.position.x;
    this->m_y = this->m_msg.info.origin.position.y;

    this->m_distSQ.clear();
    this->m_distSQ.resize(this->m_width*this->m_height,maxDist*maxDist);

    const int l(ceil(maxDist/this->m_resolution));

    for( int j = 0; j < this->m_width; j++ )
        for( int k = 0; k < this->m_height; k++ )
            //if( !this->isFeasible(j,k) )
            if( this->isObstacle(j,k) )
            {
                for( int x = MAX(0,j-l); x < MIN(this->m_width,j+l+1); x++ )
                    for( int y = MAX(0,k-l); y < MIN(this->m_height,k+l+1); y++ )
                        this->m_distSQ[x+y*this->m_width] = MIN(this->m_distSQ[x+y*this->m_width],(double)((x-j)*(x-j)+(y-k)*(y-k))*this->m_resolution*this->m_resolution);
            }

    this->m_distMap = true;

    return (true);
}

void skSubOccupancyGrid::resetDistMap()
{
    this->m_distMap = false;
}

double skSubOccupancyGrid::getDist(const sPoint2D p, const double& minDist/* = 0.0*/)
{
    if( !this->haveMsg() )
        return (-1.0);
    
    if( this->m_distMap )
        return (sqrt(this->m_distSQ[this->x2i(p.x)+this->y2i(p.y)*this->m_width]));

    // read grid data
    this->m_resolution = this->m_msg.info.resolution;
    this->m_width = this->m_msg.info.width;
    this->m_height = this->m_msg.info.height;
    this->m_x = this->m_msg.info.origin.position.x;
    this->m_y = this->m_msg.info.origin.position.y;

    const int xc(this->x2i(p.x));
    const int yc(this->y2i(p.y));
    const int range(ceil(0.5/this->m_resolution));
    int distSQ(range*range);

    for( int x = -range+1; x < range; x++ )
    {
        if( x+xc < 0 || x+xc >= this->m_width)
            continue;

        for( int y = -range+1; y < range; y++ )
        {
            if( y+yc < 0 || y+yc >= this->m_height)
                continue;
            if( x*x+y*y > range*range )
                continue;
            //if( !this->isFeasible(x+xc,y+yc) )
            if( this->isObstacle(x+xc,y+yc) )
            {
                distSQ = MIN(distSQ,x*x+y*y);
            }
        }
    }

    return (sqrt((double)distSQ)*this->m_resolution);
}

double skSubOccupancyGrid::getVoronoi(const double& x, const double& y, int& flag)
{
    if( !this->haveMsg() )
    {
        flag = -2;
        return (0.0);
    }

    // read grid data
    this->m_resolution = this->m_msg.info.resolution;
    this->m_width = this->m_msg.info.width;
    this->m_height = this->m_msg.info.height;
    this->m_x = this->m_msg.info.origin.position.x;
    this->m_y = this->m_msg.info.origin.position.y;

    const int xc(this->x2i(x));
    const int yc(this->y2i(y));

    if( 0 <= xc && xc < this->m_width && 0 <= yc && yc < this->m_height )
    {
#if 1
        if( this->isObstacle(xc,yc) )
        {
            int up(yc);
            int length_up(-1);
            while( up < this->m_height )
            {
                up++;
                if( !this->isObstacle(xc,up) )
                    break;
            }
            up--;
            if( up < this->m_height-1 )
            {
                int ceil(up);
                while( ceil < this->m_height )
                {
                    ceil++;
                    if( this->isObstacle(xc,ceil) )
                        break;
                }
                length_up = ceil - up;
            }
            int down(yc);
            int length_down(-1);
            while( down > -1 )
            {
                down--;
                if( !this->isObstacle(xc,down) )
                    break;
            }
            down++;
            if( down > 1 )
            {
                int bottom(down);
                while( bottom > -1 )
                {
                    bottom--;
                    if( this->isObstacle(xc,bottom) )
                        break;
                }
                length_down = down - bottom;
            }
            if( length_up < 0 && length_down < 0 )
            {
                flag = -1;
                return (0.0);
            }
            int ret;
            if( length_up > length_down )
            {
                ret = up + length_up/2;
            }
            else
            {
                ret = down - length_down/2;
            }
            flag = 2;
            return (this->m_y+(double)(ret)*this->m_resolution);
        }
        else
        {
            int up(yc);
            while( up < this->m_height )
            {
                up++;
                if( this->isObstacle(xc,up) )
                    break;
            }
            up--;
            int down(yc);
            while( down > -1 )
            {
                down--;
                if( this->isObstacle(xc,down) )
                    break;
            }
            down++;
            flag = 1;
            return (this->m_y+(double)((up+down)/2)*this->m_resolution);
        }
#else
        std::vector<int> s, e;
        s.clear();
        e.clear();
        bool feasible(true);
        s.push_back(0);
        for( int j = 0 ; j < this->m_height; j++ )
        {
            if( feasible )
            {
                if( this->isObstacle(j,h) )
                {
                    e.push_back(j-1);
                    feasible = false;
                }
                else if( j == this->m_height-1 )
                {
                    e.push_back(j);
                    feasible = false;
                }
            }
            else
            {
                if( !this->isObstacle(j,h) )
                {
                    s.push_back(j);
                    feasible = true;
                }
            }
        }
        if( s.size() > 0 )
        {
            int length(e[0]-s[0]);
            int ret((e[0]+s[0])/2);
            for( unsigned int j = 1; j < s.size(); j++ )
            {
                if( e[j]-s[j] > length )
                {
                    length = e[j]-s[j];
                    ret = (e[j]+s[j])/2;
                }
            }
            flag = s.size() + e.size();
            return (this->m_y+(double)ret*this->m_resolution);
        }
        else
        {
            flag = -1;
            return (0.0);
        }
#endif
    }
    else
    {
        flag = 0;
        return (0.0);
    }
}


int skSubOccupancyGrid::x2i(const double& x) const
{
    return(lround((x-this->m_x)/this->m_resolution));
    /*int i((double)round((x-this->m_x)/this->m_resolution));

    if( i < 0 )
        return (0);

    if( i >= this->m_width )
        return (this->m_width-1);

    return (i);*/
}

int skSubOccupancyGrid::y2i(const double& y) const
{
    return(lround((y-this->m_y)/this->m_resolution));
    /*const int i(lround((y-this->m_y)/this->m_resolution));

    if( i < 0 )
        return (0);

    if( i >= this->m_height )
        return (this->m_height-1);

    return (i);*/
}

bool skSubOccupancyGrid::isFeasible(const int& x, const int& y) const
{
    return (this->m_msg.data[x+y*this->m_width] == 0);
}

bool skSubOccupancyGrid::isObstacle(const int& x, const int& y) const
{
    return (this->m_msg.data[x+y*this->m_width] > 0);
}
