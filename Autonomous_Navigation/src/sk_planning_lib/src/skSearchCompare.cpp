#include "sk_planning_lib/skSearchCompare.h"

skSearchData::skSearchData()
: m_cost(0.0), m_expansion(0), m_expansionN(0), m_time(0.0), m_time_prev(0.0), m_activeCost2(false)
{};

skSearchData::skSearchData(const double& c, const unsigned int& e)
: m_cost(c), m_expansion(e), m_expansionN(0), m_time(0.0), m_time_prev(0.0), m_activeCost2(false)
{};

skSearchData::skSearchData(const double& c, const unsigned int& e, const double& t)
: m_cost(c), m_expansion(e), m_expansionN(0), m_time(t), m_time_prev(0.0), m_activeCost2(false)
{};

skSearchData::skSearchData(const double& c, const unsigned int& e, const double& t, const double& tp)
: m_cost(c), m_expansion(e), m_expansionN(0), m_time(t), m_time_prev(tp), m_activeCost2(false)
{};

skSearchData::~skSearchData()
{};

void skSearchData::activateCost2()
{
    this->m_activeCost2 = true;
}

void skSearchData::activateCost3()
{
    this->m_activeCost3 = true;
}

skSearchCompare::skSearchCompare()
: m_size(1)
{
    m_data.clear();
}

skSearchCompare::skSearchCompare(unsigned int _s)
: m_size(_s)
{
    m_data.clear();
}

skSearchCompare::~skSearchCompare()
{
    for(unsigned int j = 0; j < m_data.size(); j++)
        m_data[j].clear();
    m_data.clear();
}

bool skSearchCompare::addData(const std::vector<skSearchData> data)
{
    if( data.size() == m_size )
    {
        m_data.push_back(data);
        return (true);
    }
    return (false);
}

bool skSearchCompare::saveAsText(const char *filename)
{
  	ofstream save;
  	save.open(filename);
  	for( int j = 0; j < m_data.size(); j++ )
    {
        for( int k = 0; k < m_size; k++ )
        {
      		save << m_data[j][k].m_cost << "  ";
            if( m_data[j][k].m_activeCost2 )
                save << m_data[j][k].m_cost2 << "  ";
            save << m_data[j][k].m_expansion << "  " << m_data[j][k].m_expansionN << "  " << m_data[j][k].m_time << "  " << m_data[j][k].m_time_prev << "  ";
        }
        save <<  "\n";
    }
  	save.close();

    return (true);
}

bool skSearchCompare::saveAsTextSimple(const char *filename)
{
  	ofstream save;
  	save.open(filename);
  	for( int j = 0; j < m_data.size(); j++ )
    {
        for( int k = 0; k < m_size; k++ )
        {
      		save << m_data[j][k].m_cost << " " << m_data[j][k].m_expansion << " " << m_data[j][k].m_time << " ";
        }
        save << "\n";
    }
  	save.close();

    return (true);
}

void skSearchCompare::print() const
{
    std::vector<double> cost, exp, time, cost2;
    cost.resize(m_size-1,0.0);
    exp.resize(m_size-1,0.0);
    time.resize(m_size-1,0.0);
    cost2.resize(m_size-1,0.0);
  	for( int j = 0; j < m_data.size(); j++ )
    {
        for( int k = 1; k < m_size; k++ )
        {
            cost[k-1] += m_data[j][k].m_cost / m_data[j][0].m_cost;
            exp[k-1] += ((double)m_data[j][k].m_expansion)/((double)m_data[j][0].m_expansion);
            time[k-1] += (m_data[j][k].m_time+m_data[j][k].m_time_prev)/(m_data[j][0].m_time+m_data[j][0].m_time_prev);
            if( m_data[j][k].m_activeCost2 )
                cost2[k-1] += m_data[j][k].m_cost2 / m_data[j][0].m_cost2;
        }
    }
    for( int k = 1; k < m_size; k++ )
    {
        cost[k-1] /= (double)m_data.size();
        exp[k-1] /= (double)m_data.size();
        time[k-1] /= (double)m_data.size();
        cost2[k-1] /= (double)m_data.size();
    }
    printf("Data : %u\nCost : ", m_data.size());
    for( int k = 0; k < cost.size(); k++ )
        printf(" %10.6f", cost[k]);
    if( m_data[0][0].m_activeCost2 )
    {
        printf("\nCost2 : ", m_data.size());
        for( int k = 0; k < cost2.size(); k++ )
            printf(" %10.6f", cost2[k]);
    }
    printf("\nExp  : ");
    for( int k = 0; k < exp.size(); k++ )
        printf(" %10.6f", exp[k]);
    printf("\nTime : ");
    for( int k = 0; k < time.size(); k++ )
        printf(" %10.6f", time[k]);
    printf("\n");
}

unsigned int skSearchCompare::size() const
{
    return (this->m_data.size());
}

void skSearchCompare::compareCost() const
{
    double ratio;
    std::vector<double> min(this->m_size,std::numeric_limits<double>::max()), max(this->m_size,-1.0), mean(this->m_size,0.0);

    // analyze data
    for( unsigned int j = 0; j < this->m_data.size(); j++ )
    {
        for( unsigned int k = 1; k < this->m_size; k++ )
        {
            ratio = this->m_data[j][k].m_cost / this->m_data[j][0].m_cost;
            
            min[k] = MIN(min[k],ratio);
            max[k] = MAX(max[k],ratio);
            mean[k] += ratio;
        }
    }
    for( unsigned int k = 1; k < this->m_size; k++ )
    {
        mean[k] /= (double)this->m_data.size();
    }

    // print result
    for( unsigned int k = 1; k < this->m_size; k++ )
    {
        printf("[COST] k = %d, (min,mean,max) = (%.3f, %.3f, %.3f).\n", k, min[k], mean[k], max[k]);
    }
}

void skSearchCompare::compareCost2() const
{
    double ratio;
    std::vector<double> min(this->m_size,std::numeric_limits<double>::max()), max(this->m_size,-1.0), mean(this->m_size,0.0);

    // analyze data
    for( unsigned int j = 0; j < this->m_data.size(); j++ )
    {
        for( unsigned int k = 1; k < this->m_size; k++ )
        {
            ratio = this->m_data[j][k].m_cost2 / this->m_data[j][0].m_cost2;
            
            min[k] = MIN(min[k],ratio);
            max[k] = MAX(max[k],ratio);
            mean[k] += ratio;
        }
    }
    for( unsigned int k = 1; k < this->m_size; k++ )
    {
        mean[k] /= (double)this->m_data.size();
    }

    // print result
    for( unsigned int k = 1; k < this->m_size; k++ )
    {
        printf("[COST2] k = %d, (min,mean,max) = (%.3f, %.3f, %.3f).\n", k, min[k], mean[k], max[k]);
    }
}

void skSearchCompare::compareCost3() const
{
    double ratio;
    std::vector<double> min(this->m_size,std::numeric_limits<double>::max()), max(this->m_size,-1.0), mean(this->m_size,0.0);

    // analyze data
    for( unsigned int j = 0; j < this->m_data.size(); j++ )
    {
        for( unsigned int k = 1; k < this->m_size; k++ )
        {
            ratio = this->m_data[j][k].m_cost3 / this->m_data[j][0].m_cost3;
            
            min[k] = MIN(min[k],ratio);
            max[k] = MAX(max[k],ratio);
            mean[k] += ratio;
        }
    }
    for( unsigned int k = 1; k < this->m_size; k++ )
    {
        mean[k] /= (double)this->m_data.size();
    }

    // print result
    for( unsigned int k = 1; k < this->m_size; k++ )
    {
        printf("[COST3] k = %d, (min,mean,max) = (%.3f, %.3f, %.3f).\n", k, min[k], mean[k], max[k]);
    }
}

void skSearchCompare::compareExpand() const
{
    double ratio;
    std::vector<double> min(this->m_size,std::numeric_limits<double>::max()), max(this->m_size,-1.0), mean(this->m_size,0.0);

    // analyze data
    for( unsigned int j = 0; j < this->m_data.size(); j++ )
    {
        for( unsigned int k = 1; k < this->m_size; k++ )
        {
            ratio = ((double)this->m_data[j][k].m_expansion) / ((double)this->m_data[j][0].m_expansion);
            
            min[k] = MIN(min[k],ratio);
            max[k] = MAX(max[k],ratio);
            mean[k] += ratio;
        }
    }
    for( unsigned int k = 1; k < this->m_size; k++ )
    {
        mean[k] /= (double)this->m_data.size();
    }

    // print result
    for( unsigned int k = 1; k < this->m_size; k++ )
    {
        printf("[EXPANSION] k = %d, (min,mean,max) = (%.3f, %.3f, %.3f).\n", k, min[k], mean[k], max[k]);
    }
}

