#pragma once

#include "../utils/tp_common.h"
#include "../utils/tp_utils.h"

class GridCanvas {
public:
	GridCanvas(string name, int rows, int cols, int hsize, int csize, int hgap, int cgap);
	virtual ~GridCanvas();

    int rows();
    int cols();

    void extendRow();

    void drawTensorImage(ETensor tensor, int ndat, int nrow, int ncol);
    void drawMapImage(ETensor tensor, int ndat, int nchn, int nrow, int ncol);
    void drawFilterImage(ETensor tensor, int xchn, int ychn, int nrow, int ncol, float coef = 1.0f);

    void drawMergedMapImage(ETensor tensor, int ndat, VIntList chns, int nrow, int ncol);
    void drawMergedFilterImage(ETensor tensor, VIntList xchns, VIntList ychns, int nrow, int ncol, float coef=1.0f);

    void displayImage(int nMilliSec);
    void saveImage(string folder, string filename);

protected:
    cv::Mat m_mat;

    string m_name;
    
    int m_rows;
    int m_cols;
    int m_hsize;
    int m_csize;
    int m_hgap;
    int m_cgap;
};
