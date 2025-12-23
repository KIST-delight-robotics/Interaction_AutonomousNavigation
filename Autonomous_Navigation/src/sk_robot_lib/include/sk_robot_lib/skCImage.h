#ifndef _SK_MOBILE_LIB_C_IMAGE_H_
#define _SK_MOBILE_LIB_C_IMAGE_H_

#include <stdio.h>
#include <vector>
#include <cmath>
#include <string.h>
#include <image_transport/image_transport.h>
#include <cv_bridge/cv_bridge.h>
#include <sensor_msgs/image_encodings.h>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

#include "sk_mobile_lib/skMobileRobotCommon.h"

#define SK_IMAGE_RED (0)
#define SK_IMAGE_GREEN (1)
#define SK_IMAGE_BLUE (2)
#define SK_IMAGE_YELLOW (3)
#define SK_IMAGE_MAGENTA (4)
#define SK_IMAGE_WHITE (5)
#define SK_IMAGE_BLACK (6)
#define SK_IMAGE_GREY (7)

class CImage
{
private:
    cv_bridge::CvImage m_image;
    int m_length;
    double m_step;

    static unsigned char s_r[8], s_g[8], s_b[8];
public:
    CImage();
    CImage(const int& length, const double& step);
    ~CImage();

    void initialize(const int& length, const double& step, std::string& frame);
    void publish(ros::Publisher *pub);

    void fillColor(const char& c);
    void circle(const int& x, const int& y, const int& r, const char& c);
    void ring(const int& x, const int& y, const int& r, const int& t, const char& c);
    void line(const int& x1, const int& y1, const int& x2, const int& y2, const int& t, const int& c);
    void point(const int& x, const int& y, const char& c);
    void point(const int& x, const int& y, double r);
};

#endif