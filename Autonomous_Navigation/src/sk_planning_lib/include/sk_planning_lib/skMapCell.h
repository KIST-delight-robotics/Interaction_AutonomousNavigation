#ifndef _SK_MAP_CELL_H_
#define _SK_MAP_CELL_H_

#include "rclcpp/rclcpp.hpp"
//#include <stdio.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <ctime>

#include "nav_msgs/msg/occupancy_grid.hpp"

#include "sk_planning_lib/skGrid2D.h"
#include "sk_planning_lib/skPng.h"

using namespace std;

class skGrid2D;

class skMapCell
{
public:
    unsigned int m_maxX;
    unsigned int m_maxY;
    double m_grid_size;
    std::vector < int > m_cell; // -1 : unknown, 0 : feasible , 1 : obstacle
    
    skMapCell() : m_maxX(0), m_maxY(0), m_grid_size(1.0)
    {};

    ~skMapCell()
    {
        m_cell.clear();
    };

    bool build(const nav_msgs::msg::OccupancyGrid& grid);
    bool build(const nav_msgs::msg::OccupancyGrid& grid, const nav_msgs::msg::MapMetaData& data);
    bool setFeasible(const skGrid2D& r);
    bool setFeasible(const int& x, const int& y);
    bool setInfeasible(const skGrid2D& r);
    bool setInfeasible(const int& x, const int& y);
    bool isFeasible(const skGrid2D& r) const;
    bool isFeasible(const int& x, const int& y) const;
    skGrid2D getRandomFeasibleCell();
    std::vector<skGrid2D> getRandomBoundaries(double ratio = 0.8);

    bool save(const char *filename);
    bool load(const char *filename);
    int saveAsPng(const char *filename, const std::vector<skGrid2D>* path = NULL, const int zoom = 7);
    int saveAsPng(const char *filename, const std::vector< std::vector<skGrid2D> >& set, const std::vector<char>& color, const int zoom = 7);

    //
    skMapCell getModified(const int& N_turn_feasible, const int& N_turn_infeasible, std::vector<skGrid2D>& turned_feasible, std::vector<skGrid2D>& turned_infeasible);

    void printf();

protected:
    bool saveCell(ofstream& file);
    bool loadCell(ifstream& file);

    void addObs(skPng* png);
    void addPath(skPng* png, const std::vector<skGrid2D>* path, const char color);

    bool outRange(const skGrid2D& r) const;
    bool outRange(const int& x, const int& y) const;
};

#endif // _C_SK_MAP_CELL_H_
