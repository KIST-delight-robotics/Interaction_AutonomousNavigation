#include "sk_mobile_lib/skCImage.h"

unsigned char CImage::s_r[] = {255, 0,   0 ,  255, 255, 255, 0, 128};
unsigned char CImage::s_g[] = {0,   255, 0 ,  255, 0,   255, 0, 128};
unsigned char CImage::s_b[] = {0,   0,   255, 0,   255, 255, 0, 128};

CImage::CImage()
: m_length(0), m_step(0.0)
{
}

CImage::CImage(const int& length, const double& step)
: m_length(length), m_step(step)
{
    (m_image.image).create(2*length+1, 2*length+1, CV_8UC3);
}

CImage::~CImage()
{
}

void CImage::initialize(const int& length, const double& step, std::string& frame)
{
    m_length = length;
    m_step = step;
    (m_image.image).create(2*length+1, 2*length+1, CV_8UC3);
    m_image.header.frame_id = frame;
    m_image.encoding = std::string("rgb8");
}

void CImage::publish(ros::Publisher *pub)
{
    pub->publish(m_image.toImageMsg());
}

void CImage::fillColor(const char& c)
{
    for( int j = 0; j < m_image.image.rows; j++ )
        for( int k = 0; k < m_image.image.cols; k++ )
        {
            m_image.image.at<cv::Vec3b>(j,k) = cv::Vec3b(s_r[c],s_g[c],s_b[c]);
        }
}

void CImage::circle(const int& x, const int& y, const int& r, const char& c)
{
    //cv::circle(m_image.image, cv::Point(m_length+1-y, m_length+1-x), r, CV_RGB(s_r[c],s_g[c],s_b[c]), CV_FILLED);
    //cv::circle(m_image.image, cv::Point(m_length+1-y, m_length+1-x), r, cv::Vec3b(s_r[c],s_g[c],s_b[c]), CV_FILLED);
    cv::circle(m_image.image, cv::Point(m_length+1-y, m_length+1-x), r, cv::Vec3b(s_r[c],s_g[c],s_b[c]), cv::FILLED);
}

void CImage::ring(const int& x, const int& y, const int& r, const int& t, const char& c)
{
    cv::circle(m_image.image, cv::Point(m_length+1-y, m_length+1-x), r, cv::Vec3b(s_r[c],s_g[c],s_b[c]), t);
}

void CImage::line(const int& x1, const int& y1, const int& x2, const int& y2, const int& t, const int& c)
{
    cv::line(m_image.image, cv::Point(m_length+1-y1, m_length+1-x1), cv::Point(m_length+1-y2, m_length+1-x2), cv::Vec3b(s_r[c],s_g[c],s_b[c]), t);
}

void CImage::point(const int& x, const int& y, const char& c)
{
    m_image.image.at<cv::Vec3b>(m_length+1-x,m_length+1-y) = cv::Vec3b(s_r[c],s_g[c],s_b[c]);
}

void CImage::point(const int& x, const int& y, double r)
{
    if( r < 0.0 )
        r = 0.0;
    if( r > 1.0 )
        r = 1.0;
    int c = (int)round(255.0*r);
    m_image.image.at<cv::Vec3b>(m_length+1-x,m_length+1-y) = cv::Vec3b(255-c,c,0);
}
