#include "sk_robot_lib/skAvgFilter.h"

skAvgFilter::skAvgFilter(const int count/* = 0*/)
{
    this->m_sum = 0.0;
    this->m_data.resize(count,0.0);
    this->m_count = 0.0;
    this->m_idx = -1;
    this->m_max = count;;
    this->m_full = false;
}

skAvgFilter::~skAvgFilter()
{}

void skAvgFilter::addData(const double& data)
{
    if( this->m_max )
    {
        const int idx((this->m_idx+1)%this->m_max);
        if( this->m_full )
        {
            this->m_sum -= this->m_data[idx];
            this->m_data[idx] = data;
            this->m_sum += this->m_data[idx];
            this->m_idx = idx;
        }
        else
        {
            this->m_data[idx] = data;
            this->m_sum += this->m_data[idx];
            this->m_idx = idx;
            this->m_count += 1.0;
            if( this->m_idx+1 == this->m_max )
                this->m_full = true;
        }
    }
    else
    {
        this->m_sum += data;
        this->m_count += 1.0;
    }

}

double skAvgFilter::getAvg() const
{
    return (this->m_sum / this->m_count);
}
