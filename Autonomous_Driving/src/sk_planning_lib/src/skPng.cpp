#include "sk_planning_lib/skPng.h"

bool skPng::init(const char *filename, const int& maxX, const int& maxY, const int& zoom = 3)
{
    m_fp = fopen (filename, "wb");
    if ( !m_fp )
    {
        png_destroy_write_struct (&m_png_ptr, &m_info_ptr);
        fclose (m_fp);
        return (false);
    }

    m_png_ptr = png_create_write_struct (PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);
    if ( m_png_ptr == NULL )
    {
        fclose (m_fp);
        return (false);
    }

    m_info_ptr = png_create_info_struct (m_png_ptr);
    if ( m_info_ptr == NULL )
    {
        png_destroy_write_struct (&m_png_ptr, &m_info_ptr);
        fclose (m_fp);
        return (false);
    }

    if (setjmp (png_jmpbuf (m_png_ptr)))
    {
        png_destroy_write_struct (&m_png_ptr, &m_info_ptr);
        fclose (m_fp);
        return (false);
    }

    m_maxX = maxX;
    m_maxY = maxY;
    m_zoom = zoom;
    png_set_IHDR (m_png_ptr,
                  m_info_ptr,
                  m_zoom*m_maxX,
                  m_zoom*m_maxY,
                  8,
                  PNG_COLOR_TYPE_RGB,
                  PNG_INTERLACE_NONE,
                  PNG_COMPRESSION_TYPE_DEFAULT,
                  PNG_FILTER_TYPE_DEFAULT);
    m_row_pointers = (png_byte**)png_malloc(m_png_ptr, m_zoom*m_maxY*sizeof(png_byte *));

    for(size_t y = 0; y < m_maxY; ++y)
    {
        for(size_t i = 0; i < m_zoom; i++ )
        {
       		png_byte *row = (png_byte*)png_malloc(m_png_ptr, 3*sizeof(uint8_t)*m_maxX*m_zoom);
          	m_row_pointers[m_zoom*(m_maxY-y-1)+i] = row;
            for (size_t x = 0; x < m_maxX; ++x)
            {
           		for(size_t j = 0; j < m_zoom; j++ )
           		{
       				m_row_pointers[m_zoom*(m_maxY-y-1)+i][3*(m_zoom*x+j)] = 255; //red
	        		m_row_pointers[m_zoom*(m_maxY-y-1)+i][3*(m_zoom*x+j)+1] = 255; //green
	        		m_row_pointers[m_zoom*(m_maxY-y-1)+i][3*(m_zoom*x+j)+2] = 255; //blue
	            }
	        }
		}
	}

    return (true);
}

bool skPng::close()
{
    png_init_io (m_png_ptr, m_fp);
    png_set_rows (m_png_ptr, m_info_ptr, m_row_pointers);
    png_write_png (m_png_ptr, m_info_ptr, PNG_TRANSFORM_IDENTITY, NULL);

    for (size_t y = 0; y < m_maxY*m_zoom; y++)
    {
        png_free(m_png_ptr, m_row_pointers[y]);
    }
    png_free(m_png_ptr, m_row_pointers);
    png_destroy_write_struct (&m_png_ptr, &m_info_ptr);
    fclose (m_fp);

    return (true);
}

bool skPng::fillBG(const char& c)
{
    switch( c )
    {
        case 'r' :
        case 'R' :
            return fillBG(255, 0, 0);
            break;
        case 'g' :
        case 'G' :
            return fillBG(0, 255, 0);
            break;
        case 'b' :
        case 'B' :
            return fillBG(0, 0, 255);
            break;
        case 'k' :
        case 'K' :
            return fillBG(0, 0, 0);
            break;
        case 'w' :
        case 'W' :
            return fillBG(255, 255, 255);
            break;
        case 'm' :
        case 'M' :
            return fillBG(255, 0, 255);
            break;
        case 'c' :
        case 'C' :
            return fillBG(0, 255, 255);
            break;
        case 'y' :
        case 'Y' :
            return fillBG(255, 255, 0);
            break;
    }
    return (false);
}

bool skPng::fillBG(const uint8_t& r, const uint8_t& g, const uint8_t& b)
{
    for(size_t y = 0; y < m_maxY; ++y)
    {
        for(size_t i = 0; i < m_zoom; i++ )
        {
            for (size_t x = 0; x < m_maxX; ++x)
            {
           		for(size_t j = 0; j < m_zoom; j++ )
           		{
       				m_row_pointers[m_zoom*(m_maxY-y-1)+i][3*(m_zoom*x+j)] = r; //red
	        		m_row_pointers[m_zoom*(m_maxY-y-1)+i][3*(m_zoom*x+j)+1] = g; //green
	        		m_row_pointers[m_zoom*(m_maxY-y-1)+i][3*(m_zoom*x+j)+2] = b; //blue
	            }
	        }
		}
	}
    return (true);
}

bool skPng::fillCell(const int& x, const int& y, const char& c)
{
    switch( c )
    {
        case 'r' :
        case 'R' :
            return fillCell(x, y, 255, 0, 0);
            break;
        case 'g' :
        case 'G' :
            return fillCell(x, y, 0, 255, 0);
            break;
        case 'b' :
        case 'B' :
            return fillCell(x, y, 0, 0, 255);
            break;
        case 'k' :
        case 'K' :
            return fillCell(x, y, 0, 0, 0);
            break;
        case 'w' :
        case 'W' :
            return fillCell(x, y, 255, 255, 255);
            break;
        case 'm' :
        case 'M' :
            return fillCell(x, y, 255, 0, 255);
            break;
        case 'c' :
        case 'C' :
            return fillCell(x, y, 0, 255, 255);
            break;
        case 'y' :
        case 'Y' :
            return fillCell(x, y, 255, 255, 0);
            break;
    }
    return (false);
}

bool skPng::fillCell(const int& x, const int& y, const uint8_t& r, const uint8_t& g, const uint8_t& b)
{
    if( x < 0 || x>= m_maxX || y < 0 || y >= m_maxY )
        return (false);

    for(size_t i = 0; i < m_zoom; i++ )
   		for(size_t j = 0; j < m_zoom; j++ )
   		{
			m_row_pointers[m_zoom*(m_maxY-y-1)+i][3*(m_zoom*x+j)] = r; //red
      		m_row_pointers[m_zoom*(m_maxY-y-1)+i][3*(m_zoom*x+j)+1] = g; //green
       		m_row_pointers[m_zoom*(m_maxY-y-1)+i][3*(m_zoom*x+j)+2] = b; //blue
		}

    return (true);    
}

bool skPng::fillCircle(const int& x, const int& y, const int& l, const char& c)
{
    switch( c )
    {
        case 'r' :
        case 'R' :
            return fillCircle(x, y, l, 255, 0, 0);
            break;
        case 'g' :
        case 'G' :
            return fillCircle(x, y, l, 0, 255, 0);
            break;
        case 'b' :
        case 'B' :
            return fillCircle(x, y, l, 0, 0, 255);
            break;
        case 'k' :
        case 'K' :
            return fillCircle(x, y, l, 0, 0, 0);
            break;
        case 'w' :
        case 'W' :
            return fillCircle(x, y, l, 255, 255, 255);
            break;
        case 'm' :
        case 'M' :
            return fillCircle(x, y, l, 255, 0, 255);
            break;
        case 'c' :
        case 'C' :
            return fillCircle(x, y, l, 0, 255, 255);
            break;
        case 'y' :
        case 'Y' :
            return fillCircle(x, y, l, 255, 255, 0);
            break;
    }
    return (false);
}

bool skPng::fillCircle(const int& xc, const int& yc, const int& l, const uint8_t& r, const uint8_t& g, const uint8_t& b)
{
    int xl = MAX(0,xc-l-1);
    int xu = MIN(xc+l+2,m_maxX);
    int yl = MAX(0,yc-l-1);
    int yu = MIN(yc+l+2,m_maxY);

    for(size_t y = 0; y < m_maxY; ++y)
    {
        for(size_t x = 0; x < m_maxX; ++x)
        {
            if( (x-xc)*(x-xc)+(y-yc)*(y-yc) > l*l )
                continue;
            for(size_t i = 0; i < m_zoom; i++ )
            {
           		for(size_t j = 0; j < m_zoom; j++ )
           		{
       				m_row_pointers[m_zoom*(m_maxY-y-1)+i][3*(m_zoom*x+j)] = r; //red
	        		m_row_pointers[m_zoom*(m_maxY-y-1)+i][3*(m_zoom*x+j)+1] = g; //green
	        		m_row_pointers[m_zoom*(m_maxY-y-1)+i][3*(m_zoom*x+j)+2] = b; //blue
	            }
	        }
		}
	}
    return (true);
}

bool skPng::addBox(const int& xl, const int& xu, const int& yl, const int& yu, const char& c)
{
    switch( c )
    {
        case 'r' :
        case 'R' :
            return addBox(xl, xu, yl, yu, 255, 0, 0);
            break;
        case 'g' :
        case 'G' :
            return addBox(xl, xu, yl, yu, 0, 255, 0);
            break;
        case 'b' :
        case 'B' :
            return addBox(xl, xu, yl, yu, 0, 0, 255);
            break;
        case 'k' :
        case 'K' :
            return addBox(xl, xu, yl, yu, 0, 0, 0);
            break;
        case 'w' :
        case 'W' :
            return addBox(xl, xu, yl, yu, 255, 255, 255);
            break;
        case 'm' :
        case 'M' :
            return addBox(xl, xu, yl, yu, 255, 0, 255);
            break;
        case 'c' :
        case 'C' :
            return addBox(xl, xu, yl, yu, 0, 255, 255);
            break;
        case 'y' :
        case 'Y' :
            return addBox(xl, xu, yl, yu, 255, 255, 0);
            break;
    }
    return (false);
}

bool skPng::addBox(const int& xl, const int& xu, const int& yl, const int& yu, const uint8_t& r, const uint8_t& g, const uint8_t& b)
{
    int xi = MAX(0,MIN(xl,xu));
    int xf = MIN(MAX(xl,xu)+1,m_maxX);
    int yi = MAX(0,MIN(yl,yu));
    int yf = MIN(MAX(yl,yu)+1,m_maxY);

    for(size_t i = 0; i < m_zoom; i++ )
    {
   		for(size_t j = 0; j < m_zoom; j++ )
   		{
            for(size_t y = yi; y < yf; ++y)
            {
   				m_row_pointers[m_zoom*(m_maxY-y-1)+i][3*(m_zoom*xi+j)] = r; //red
          		m_row_pointers[m_zoom*(m_maxY-y-1)+i][3*(m_zoom*xi+j)+1] = g; //green
        		m_row_pointers[m_zoom*(m_maxY-y-1)+i][3*(m_zoom*xi+j)+2] = b; //blue
   				m_row_pointers[m_zoom*(m_maxY-y-1)+i][3*(m_zoom*xf+j)] = r; //red
          		m_row_pointers[m_zoom*(m_maxY-y-1)+i][3*(m_zoom*xf+j)+1] = g; //green
        		m_row_pointers[m_zoom*(m_maxY-y-1)+i][3*(m_zoom*xf+j)+2] = b; //blue
	        }
            for(size_t x = xi; x < xf; ++x)
            {
   				m_row_pointers[m_zoom*(m_maxY-yi-1)+i][3*(m_zoom*x+j)] = r; //red
          		m_row_pointers[m_zoom*(m_maxY-yi-1)+i][3*(m_zoom*x+j)+1] = g; //green
        		m_row_pointers[m_zoom*(m_maxY-yi-1)+i][3*(m_zoom*x+j)+2] = b; //blue
   				m_row_pointers[m_zoom*(m_maxY-yf-1)+i][3*(m_zoom*x+j)] = r; //red
          		m_row_pointers[m_zoom*(m_maxY-yf-1)+i][3*(m_zoom*x+j)+1] = g; //green
        		m_row_pointers[m_zoom*(m_maxY-yf-1)+i][3*(m_zoom*x+j)+2] = b; //blue
	        }
		}
	}
    return (true);
}

bool skPng::fillBox(const int& xl, const int& xu, const int& yl, const int& yu, const char& c)
{
    switch( c )
    {
        case 'r' :
        case 'R' :
            return fillBox(xl, xu, yl, yu, 255, 0, 0);
            break;
        case 'g' :
        case 'G' :
            return fillBox(xl, xu, yl, yu, 0, 255, 0);
            break;
        case 'b' :
        case 'B' :
            return fillBox(xl, xu, yl, yu, 0, 0, 255);
            break;
        case 'k' :
        case 'K' :
            return fillBox(xl, xu, yl, yu, 0, 0, 0);
            break;
        case 'w' :
        case 'W' :
            return fillBox(xl, xu, yl, yu, 255, 255, 255);
            break;
        case 'm' :
        case 'M' :
            return fillBox(xl, xu, yl, yu, 255, 0, 255);
            break;
        case 'c' :
        case 'C' :
            return fillBox(xl, xu, yl, yu, 0, 255, 255);
            break;
        case 'y' :
        case 'Y' :
            return fillBox(xl, xu, yl, yu, 255, 255, 0);
            break;
    }
    return (false);
}

bool skPng::fillBox(const int& xl, const int& xu, const int& yl, const int& yu, const uint8_t& r, const uint8_t& g, const uint8_t& b)
{
    int xi = MAX(0,MIN(xl,xu));
    int xf = MIN(MAX(xl,xu)+1,m_maxX);
    int yi = MAX(0,MIN(yl,yu));
    int yf = MIN(MAX(yl,yu)+1,m_maxY);

    for(size_t y = yi; y < yf; ++y)
    {
        for(size_t x = xi; x < xf; ++x)
        {
            for(size_t i = 0; i < m_zoom; i++ )
            {
   		        for(size_t j = 0; j < m_zoom; j++ )
   		        {
   		    		m_row_pointers[m_zoom*(m_maxY-y-1)+i][3*(m_zoom*x+j)] = r; //red
              		m_row_pointers[m_zoom*(m_maxY-y-1)+i][3*(m_zoom*x+j)+1] = g; //green
            		m_row_pointers[m_zoom*(m_maxY-y-1)+i][3*(m_zoom*x+j)+2] = b; //blue
                }
	        }
		}
	}
    return (true);
}
