#ifndef _SK_SEARCH_COMPARE_H_
#define _SK_SEARCH_COMPARE_H_

#include <iostream>
#include <fstream>
#include <vector>
#include <sk_planning_lib/skPlanningCommon.h>

using namespace std;

class skSearchData
{
public:
    double m_cost;
    double m_cost2;
    bool m_activeCost2;
    double m_cost3;
    bool m_activeCost3;
    unsigned int m_expansion;
    unsigned int m_expansionN;
    double m_time;
    double m_time_prev;

    skSearchData();
    skSearchData(const double& c, const unsigned int& e);
    skSearchData(const double& c, const unsigned int& e, const double& t);
    skSearchData(const double& c, const unsigned int& e, const double& t, const double& tp);
    ~skSearchData();
    void activateCost2();
    void activateCost3();
};

class skSearchCompare
{
private:
    unsigned int m_size;
    std::vector< std::vector<skSearchData> > m_data;

public:
    skSearchCompare();
    skSearchCompare(unsigned int _s);
    ~skSearchCompare();

    bool addData(const std::vector<skSearchData> data);
    bool saveAsText(const char *filename);
    bool saveAsTextSimple(const char *filename);
    void print() const;
    unsigned int size() const;
    void compareCost() const;
    void compareCost2() const;
    void compareCost3() const;
    void compareExpand() const;
};

#endif // _CALL_HEURISTIC_H_