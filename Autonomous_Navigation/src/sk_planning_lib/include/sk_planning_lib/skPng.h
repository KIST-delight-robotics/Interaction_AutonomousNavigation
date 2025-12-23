#ifndef _SK_PNG_H_
#define _SK_PNG_H_

//#include <stdio.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <png.h>
#include <stdint.h>
//#include "classList.h"
#include "sk_planning_lib/skPlanningCommon.h"

using namespace std;

class skPng
{
private:
    FILE *m_fp;
    png_structp m_png_ptr;
    png_infop m_info_ptr;
    png_byte **m_row_pointers;
    int m_maxX, m_maxY, m_zoom;

public:
    skPng() :
        m_png_ptr(NULL), m_info_ptr(NULL), m_row_pointers(NULL)
    {};
    ~skPng()
    {};

    bool init(const char *filename, const int& maxX, const int& maxY, const int& zoom);
    bool close();
    bool fillBG(const char& c);
    bool fillBG(const uint8_t& r, const uint8_t& g, const uint8_t& b);

    bool fillCell(const int& x, const int& y, const char& c);
    bool fillCell(const int& x, const int& y, const uint8_t& r, const uint8_t& g, const uint8_t& b);

    bool fillCircle(const int& x, const int& y, const int& l, const char& c);
    bool fillCircle(const int& x, const int& y, const int& l, const uint8_t& r, const uint8_t& g, const uint8_t& b);

    bool addBox(const int& xl, const int& xu, const int& yl, const int& yu, const char& c);
    bool addBox(const int& xl, const int& xu, const int& yl, const int& yu, const uint8_t& r, const uint8_t& g, const uint8_t& b);

    bool fillBox(const int& xl, const int& xu, const int& yl, const int& yu, const char& c);
    bool fillBox(const int& xl, const int& xu, const int& yl, const int& yu, const uint8_t& r, const uint8_t& g, const uint8_t& b);
};

#endif // _C_SK_PNG_H_
