#ifndef _SK_MAP_DIST_H_
#define _SK_MAP_DIST_H_

//#include <stdio.h>
#include <iostream>
#include <fstream>
#include <vector>
//#include "classList.h"
//#include "sk_planning_lib/skMacro.h"
//#include "sk_planning_lib/skGrid2.h"
//#include "sk_planning_lib/skPng.h"
#include "sk_planning_lib/skMapCell.h"
//#include "nav_msgs/msg/OccupancyGrid.h"
//#include "geometry_msgs/msg/Pose2D.h"

using namespace std;

//class skGrid2D;

class skMapDist : public skMapCell
{
public:
    double m_maxDist;
    std::vector < double > m_distSQ; // dist to closest obstacle or unknown cell
    
    skMapDist() : m_maxDist(1.0)
    {        
    }

    ~skMapDist()
    {
        this->m_distSQ.clear();
    }

    bool build(const nav_msgs::msg::OccupancyGrid& grid);
    bool build(const nav_msgs::msg::OccupancyGrid& grid, const nav_msgs::msg::MapMetaData& data);
    bool build(const skMapCell& cell, const double& maxDist = 10.0);

    using skMapCell::isFeasible;  // Expose base class methods
    bool isFeasible(const skGrid2D& r, const double& margin) const;
    bool isFeasible(const int& x, const int& y, const double& margin) const;
    //bool isVisible(const skGrid2& r1, const skGrid2& r2, const double& margin);
    //bool isVisible(const skGrid2& r1, const skGrid2& r2, const double& margin, double& minDist);
    double getDist(const skGrid2D& r) const;
    double getDist(const int& x, const int& y) const;
    bool save(const char *filename);
    bool load(const char *filename);
    int saveAsPng(const char *filename, const std::vector<skGrid2D>* path = NULL, const int zoom = 7);
    int saveAsPng(const char *filename, const std::vector< std::vector<skGrid2D> >& set, const std::vector<char>& color, const int zoom = 7);

protected:
    bool saveDist(ofstream& file);
    bool loadDist(ifstream& file);

    void addDistGradient(skPng *png);
    bool updateDist(const double maxDist = -1.0);
};

#endif // _C_SK_GRID_H_
