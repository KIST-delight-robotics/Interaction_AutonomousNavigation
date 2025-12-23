#include "sk_planning_lib/skMapCell.h"
#include "sk_planning_lib/skGrid2D.h"
#include "stdio.h"

bool skMapCell::outRange(const skGrid2D& r) const
{
    return (this->outRange(r.m_x, r.m_y));
}

bool skMapCell::outRange(const int& x, const int& y) const
{
    if( x < 0 || x >= m_maxX || y < 0 || y >= m_maxY )
        return (true);
    return (false);
}

bool skMapCell::setFeasible(const skGrid2D& r)
{
    return (this->setFeasible(r.m_x, r.m_y));
}

bool skMapCell::setFeasible(const int& x, const int& y)
{
    if( this->outRange(x,y) )
        return (false);
    this->m_cell[x*m_maxY+y] = SK_CELL_FEASIBLE;

    return(true);
}

bool skMapCell::setInfeasible(const skGrid2D& r)
{
    return (this->setInfeasible(r.m_x, r.m_y));
}

bool skMapCell::setInfeasible(const int& x, const int& y)
{
    if( this->outRange(x,y) )
        return (false);
    this->m_cell[x*m_maxY+y] = SK_CELL_INFEASIBLE;

    return(true);
}

bool skMapCell::isFeasible(const skGrid2D& r) const
{
    return (this->isFeasible(r.m_x, r.m_y));
}

bool skMapCell::isFeasible(const int& x, const int& y) const
{
    if( this->outRange(x,y) )
        return (false);
    return (this->m_cell[x*m_maxY+y] == SK_CELL_FEASIBLE );
}

bool skMapCell::save(const char *filename)
{
//    int j;
    ofstream OutFile;
    OutFile.open(filename, ios::out | ios::binary);
    this->saveCell(OutFile);
/*    OutFile.write( (char*)&m_maxX, sizeof(int));
    OutFile.write( (char*)&m_maxY, sizeof(int));
    int temp;
    for( j = 0; j < m_maxX*m_maxY; j++ )
    {
        temp = m_cell[j];
        OutFile.write( (char*)&temp, sizeof(int));
    }*/
    OutFile.close();

    return (true);
}

bool skMapCell::saveCell(ofstream& file)
{
    file.write( (char*)&this->m_maxX, sizeof(int));
    file.write( (char*)&this->m_maxY, sizeof(int));
    file.write( (char*)&this->m_grid_size, sizeof(double));
    int temp;
    for( unsigned int j = 0; j < this->m_maxX*this->m_maxY; j++ )
    {
        temp = this->m_cell[j];
        file.write( (char*)&temp, sizeof(int));
    }

    return (true);
}

bool skMapCell::load(const char *filename)
{
//    int j;
    ifstream InFile;
    InFile.open(filename, ios::in | ios::binary);
    this->loadCell(InFile);
/*    InFile.read( (char*)&m_maxX, sizeof(int));
    InFile.read( (char*)&m_maxY, sizeof(int));
    m_cell.resize(m_maxX*m_maxY);
    int temp;
    for( j = 0; j < m_maxX*m_maxY; j++ )
    {
        InFile.read( (char*)&temp, sizeof(int));
        m_cell[j] = temp;
    }*/
    InFile.close();
}

bool skMapCell::loadCell(ifstream& file)
{
    file.read( (char*)&this->m_maxX, sizeof(int));
    file.read( (char*)&this->m_maxY, sizeof(int));
    file.read( (char*)&this->m_grid_size, sizeof(double));
    this->m_cell.resize(this->m_maxX*this->m_maxY);
    int temp;
    for( unsigned int j = 0; j < this->m_maxX*this->m_maxY; j++ )
    {
        file.read( (char*)&temp, sizeof(int));
        this->m_cell[j] = temp;
    }

    return (true);
}

int skMapCell::saveAsPng(const char *filename, const std::vector<skGrid2D>* path, const int zoom)
{
//    uint8_t r, g, b;

    skPng png;
    
    png.init(filename, m_maxX, m_maxY, zoom);
    png.fillBG('w');

    this->addObs(&png);
/*    for( int y = 0; y < m_maxY; ++y)
    {
        for (int x = 0; x < m_maxX; ++x)
        {
            if( m_cell[x*m_maxY+y] == SK_CELL_FEASIBLE )
            {
                r = 255;
                g = 255;
                b = 255;
            }
            else if( m_cell[x*m_maxY+y] == SK_CELL_UNKNOWN )
            {
                r = 169;
                g = 169;
                b = 169;
            }
            else
            {
   				r = 0;
           		g = 0;
           		b = 0;
            }
            png.fillCell(x,y,r,g,b);
		}
	}
*/
    if( path )
    {
        this->addPath(&png, path, 'g');
/*        for( unsigned int j = 0; j < path->size(); j++ )
        {
            png.fillCell(path->at(j).m_x, path->at(j).m_y, 'g');
        }*/
    }
    png.close();

    return (0);
}

int skMapCell::saveAsPng(const char *filename, const std::vector< std::vector<skGrid2D> >& set, const std::vector<char>& color, const int zoom/* = 7*/)
{
//    uint8_t r, g, b;

    skPng png;
    
    png.init(filename, m_maxX, m_maxY, zoom);
    png.fillBG('w');

    this->addObs(&png);
/*    for( int y = 0; y < m_maxY; ++y)
    {
        for (int x = 0; x < m_maxX; ++x)
        {
            if( m_cell[x*m_maxY+y] == SK_CELL_FEASIBLE )
            {
                r = 255;
                g = 255;
                b = 255;
            }
            else if( m_cell[x*m_maxY+y] == SK_CELL_UNKNOWN )
            {
                r = 169;
                g = 169;
                b = 169;
            }
            else
            {
   				r = 0;
           		g = 0;
           		b = 0;
            }
            png.fillCell(x,y,r,g,b);
		}
	}*/
    for( unsigned int j = 0; j < set.size(); j++ )
    {
        this->addPath(&png, &(set[j]), color[j]);
        /*for( unsigned int k = 0; k < set[j].size(); k++ )
        {
            png.fillCell(set[j][k].m_x, set[j][k].m_y, color[j]);
        }*/
    }
    png.close();

    return (0);
}

void skMapCell::addObs(skPng* png)
{
    uint8_t r, g, b;

    for( unsigned int y = 0; y < this->m_maxY; ++y)
    {
        for( unsigned int x = 0; x < this->m_maxX; ++x)
        {
            if( this->m_cell[x*this->m_maxY+y] == SK_CELL_FEASIBLE )
            {
                r = 255;
                g = 255;
                b = 255;
            }
            else if( m_cell[x*m_maxY+y] == SK_CELL_UNKNOWN )
            {
                r = 169;
                g = 169;
                b = 169;
            }
            else
            {
   				r = 0;
           		g = 0;
           		b = 0;
            }
            png->fillCell(x,y,r,g,b);
		}
	}
}

void skMapCell::addPath(skPng* png, const std::vector<skGrid2D>* path, const char color)
{
    for( unsigned int j = 0; j < path->size(); j++ )
    {
        png->fillCell(path->at(j).x(), path->at(j).y(), color);
    }
}

bool skMapCell::build(const nav_msgs::msg::OccupancyGrid& grid)
{
    return build(grid, grid.info);
}

bool skMapCell::build(const nav_msgs::msg::OccupancyGrid& grid, const nav_msgs::msg::MapMetaData& data)
{
    m_maxX = data.width;
    m_maxY = data.height;
    m_cell.resize(m_maxX*m_maxY, 0);
    for( unsigned int j = 0; j < m_maxX; j++ )
        for( unsigned int k = 0; k < m_maxY; k++ )
        {
            if( grid.data[k*m_maxX+j] < 0 )
                m_cell[j*m_maxY+k] = SK_CELL_UNKNOWN;
            else if( grid.data[k*m_maxX+j] > 65 )
                m_cell[j*m_maxY+k] = SK_CELL_INFEASIBLE;
            else
                m_cell[j*m_maxY+k] = SK_CELL_FEASIBLE;
        }

    return (true);
}

skGrid2D skMapCell::getRandomFeasibleCell()
{
    skGrid2D cell;

    std::srand(std::time(NULL)); // use current time as seed for random generator
    do
    {
        cell.m_x = (std::rand())%m_maxX;
        cell.m_y = (std::rand())%m_maxY;
    } while (!isFeasible(cell));

    return (cell);
}

std::vector<skGrid2D> skMapCell::getRandomBoundaries(double ratio)
{
    std::vector< skGrid2D > cells, ret;
    cells.clear();
    ret.clear();
    skGrid2D cell;

    for( int x = 0; x < m_maxX; x++ )
    {
        cell.m_x = x;
        for( int y = 0; y < m_maxY; y++ )
        {
            cell.m_y = y;
            if( isFeasible(cell) )
                cells.push_back(cell);
        }
    }

    ratio = MIN(ratio,0.9);
    ratio = MAX(ratio,0.1);
    double l = ratio*((double)(m_maxX+m_maxY));
    int s, g;
    //std::srand(std::time(NULL)); // use current time as seed for random generator
    do
    {
        s = (std::rand())%cells.size();
        g = (std::rand())%cells.size();
    //} while (cells[s].dist4To(cells[g]) < l);
    } while ((double)(abs(cells[s].m_x-cells[g].m_x)+abs(cells[s].m_y-cells[g].m_y))*this->m_grid_size < l);

    ret.push_back(cells[s]);
    ret.push_back(cells[g]);

    return (ret);
}

skMapCell skMapCell::getModified(const int& N_turn_feasible, const int& N_turn_infeasible, std::vector<skGrid2D>& turned_feasible, std::vector<skGrid2D>& turned_infeasible)
{
    skMapCell ret(*this);
    skGrid2D grid;
    int r;

    // clear returning vectors;
    turned_feasible.clear();
    turned_infeasible.clear();

    std::srand(std::time(NULL)); // use current time as seed for random generator

    // turn N cells feasible
    std::vector<int> candi_x, candi_y;
    candi_x.clear();
    candi_y.clear();
    for( int x = 1; x+1 < ret.m_maxX; x++ )
        for( int y = 1; y+1 < ret.m_maxY; y++ )
        {
            //if( (!this->isFeasible(x,y)) && ((this->isFeasible(x-1,y)&&this->isFeasible(x+1,y))||(this->isFeasible(x,y-1)&&this->isFeasible(x,y+1))) )
            if( (!this->isFeasible(x,y)) && this->isFeasible(x-1,y) && this->isFeasible(x+1,y) && (!this->isFeasible(x,y-1)) && (!this->isFeasible(x,y+1)) )
            {
                candi_x.push_back(x); 
                candi_y.push_back(y); 
            }
            else if( (!this->isFeasible(x,y)) && this->isFeasible(x,y-1) && this->isFeasible(x,y+1) && (!this->isFeasible(x-1,y)) && (!this->isFeasible(x+1,y)) )
            {
                candi_x.push_back(x); 
                candi_y.push_back(y); 
            }
        }
    for( int j = 0; j < MIN(N_turn_feasible, candi_x.size()); j++ )
    {
        r = (std::rand())%candi_x.size();
        grid.m_x = candi_x[r];
        grid.m_y = candi_y[r];
        ret.m_cell[grid.m_x*ret.m_maxY+grid.m_y] = SK_CELL_FEASIBLE;
        candi_x.erase(candi_x.begin()+r);
        candi_y.erase(candi_y.begin()+r);
        turned_feasible.push_back(grid);
    }

    // turn N cells infeasible
    candi_x.clear();
    candi_y.clear();
    for( int x = 3; x+3 < ret.m_maxX; x++ )
        for( int y = 3; y+3 < ret.m_maxY; y++ )
        {
            if( this->isFeasible(x,y) && this->isFeasible(x-1,y) && this->isFeasible(x+1,y) && (!this->isFeasible(x,y-1)) && (!this->isFeasible(x,y+1)) )
            {
                candi_x.push_back(x); 
                candi_y.push_back(y); 
            }
            else if( this->isFeasible(x,y) && this->isFeasible(x,y-1) && this->isFeasible(x,y+1) && (!this->isFeasible(x-1,y)) && (!this->isFeasible(x+1,y)) )
            {
                candi_x.push_back(x); 
                candi_y.push_back(y); 
            }
        }
    for( int j = 0; j < MIN(N_turn_infeasible, candi_x.size()); j++ )
    {
        r = (std::rand())%candi_x.size();
        grid.m_x = candi_x[r];
        grid.m_y = candi_y[r];
        ret.m_cell[grid.m_x*ret.m_maxY+grid.m_y] = SK_CELL_INFEASIBLE;
        candi_x.erase(candi_x.begin()+r);
        candi_y.erase(candi_y.begin()+r);
        turned_infeasible.push_back(grid);
    }

    return (ret);
}

void skMapCell::printf()
{
    for( int x = 0; x < m_maxX; x++ )
    {
        for( int y = 0; y < m_maxY; y++)
            if( m_cell[x*m_maxY+y] == SK_CELL_FEASIBLE )
                std::printf(" ");
            else if( m_cell[x*m_maxY+y] == SK_CELL_INFEASIBLE )
                std::printf("#");
            else
                std::printf("?");
        std::printf("\n");
    }
}
