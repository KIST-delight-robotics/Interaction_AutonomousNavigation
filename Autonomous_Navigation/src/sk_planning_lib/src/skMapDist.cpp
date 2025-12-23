#include "sk_planning_lib/skMapDist.h"

bool skMapDist::build(const nav_msgs::msg::OccupancyGrid& grid)
{
    if( skMapCell::build(grid) )
        return (this->updateDist());

    return (false);
}

bool skMapDist::build(const nav_msgs::msg::OccupancyGrid& grid, const nav_msgs::msg::MapMetaData& data)
{
    if( skMapCell::build(grid, data) )
    {
        this->m_grid_size = data.resolution;
        return (this->updateDist());
    }

    return (false);
}

bool skMapDist::build(const skMapCell& cell, const double& maxDist/* = 10.0*/)
{
    this->m_grid_size = 1.0;
    this->m_maxX = cell.m_maxX;
    this->m_maxY = cell.m_maxY;
    this->m_cell.resize(m_maxX*m_maxY, 0);
    for( int j = 0; j < m_cell.size(); j++ )
        this->m_cell[j] = cell.m_cell[j];
    
    this->m_maxDist = maxDist;

    return (this->updateDist());
}

bool skMapDist::isFeasible(const skGrid2D& r, const double& margin) const
{
    return (this->isFeasible(r.x(), r.y(), margin));
}

bool skMapDist::isFeasible(const int& x, const int& y, const double& margin) const
{
    if( this->outRange(x,y) )
        return (false);

    if( this->skMapCell::isFeasible(x,y) )
        return (!(this->m_distSQ[x*this->m_maxY+y] < (margin*margin)) );

    return (false);
}

double skMapDist::getDist(const skGrid2D& r) const
{
    return (this->getDist(r.m_x, r.m_y));
}

double skMapDist::getDist(const int& x, const int& y) const
{
    if( this->outRange(x,y) )
        return (-1.0);

    return (sqrt(this->m_distSQ[x*m_maxY+y]));
}

bool skMapDist::save(const char *filename)
{
    int j;
    ofstream OutFile;
    OutFile.open(filename, ios::out | ios::binary);
    this->saveCell(OutFile);
    this->saveDist(OutFile);
    OutFile.close();

    return (true);
}

bool skMapDist::saveDist(ofstream& file)
{
    file.write( (char*)&this->m_maxDist, sizeof(double));
    double temp;
    for( unsigned int j = 0; j < this->m_maxX*this->m_maxY; j++ )
    {
        temp = this->m_distSQ[j];
        file.write( (char*)&temp, sizeof(double));
    }

    return (true);
}

bool skMapDist::load(const char *filename)
{
    int j;
    ifstream InFile;
    InFile.open(filename, ios::in | ios::binary);
    this->loadCell(InFile);
    this->loadDist(InFile);
    InFile.close();
    return (true);
}

bool skMapDist::loadDist(ifstream& file)
{
    file.read( (char*)&this->m_maxDist, sizeof(double));
    this->m_distSQ.resize(this->m_maxX*this->m_maxY);
    double temp;
    for(  unsigned int j = 0; j < this->m_maxX*this->m_maxY; j++ )
    {
        file.read( (char*)&temp, sizeof(double));
        this->m_distSQ[j] = temp;
    }

    return (false);
}

int skMapDist::saveAsPng(const char *filename, const std::vector<skGrid2D>* path, const int zoom)
{
    skPng png;
    
    png.init(filename, m_maxX, m_maxY, zoom);
    png.fillBG('w');
    this->addObs(&png);
    this->addDistGradient(&png);

    if( path )
    {
        this->addPath(&png, path, 'g');
    }

    png.close();

    return (0);
}
int skMapDist::saveAsPng(const char *filename, const std::vector< std::vector<skGrid2D> >& set, const std::vector<char>& color, const int zoom/* = 7*/)
{
    uint8_t r, g, b;

    skPng png;
    
    png.init(filename, m_maxX, m_maxY, zoom);
    png.fillBG('w');
    this->addObs(&png);
    this->addDistGradient(&png);

    // plot paths
    for( unsigned int j = 0; j < set.size(); j++ )
    {
        this->addPath(&png, &(set[j]), color[j]);
    }

    png.close();

    return (0);
}

void skMapDist::addDistGradient(skPng *png)
{
    uint8_t r, g, b;

    double ratio;
    for( unsigned int y = 0; y < m_maxY; ++y)
    {
        for( unsigned int x = 0; x < m_maxX; ++x)
        {
            if( this->m_cell[x*m_maxY+y] == 0 )
            {
                ratio = MIN(1.0, sqrt(this->m_distSQ[x*this->m_maxY+y])/this->m_maxDist);
                r = 255;
                g = (int)round(255.0*ratio);
                b = g;
                png->fillCell(x,y,r,g,b);
    		}
	    }
    }
}

bool skMapDist::updateDist(const double maxDist/* = -1.0*/)
{
    // Some parameters to be determined
    //m_maxDist = 1.5;
    if( maxDist > 0.0 )
        this->m_maxDist = maxDist;

    const double maxDistSQ(this->m_maxDist*this->m_maxDist);
    // initialize the map
    this->m_distSQ.clear();
    this->m_distSQ.resize(this->m_maxX*this->m_maxY, maxDistSQ);
    // find constants
    int maxR((int)ceil(this->m_maxDist/this->m_grid_size)+2);

    int j, k, x, y;
    double distSQ;
    for( j = 0; j < this->m_maxX; j++ )
        for( k = 0; k < this->m_maxY; k++ )
            if( this->m_cell[j*this->m_maxY+k] > 0 )
            {
                for( x = MAX(0,j-maxR); x < MIN(m_maxX,j+maxR+1); x++ )
                    for( y = MAX(0,k-maxR); y < MIN(m_maxY,k+maxR+1); y++ )
                    {
                        distSQ = (double)((j-x)*(j-x)+(k-y)*(k-y));
                        distSQ *= this->m_grid_size*this->m_grid_size;
                        //if( distSQ < maxDistSQ )
                        //    m_distSQ[x*m_maxY+y] = MIN(m_distSQ[x*m_maxY+y], distSQ);
                        if( distSQ < this->m_distSQ[x*this->m_maxY+y] )
                        this->m_distSQ[x*this->m_maxY+y] = distSQ;
                    }
                        
            }

    return (true);
}