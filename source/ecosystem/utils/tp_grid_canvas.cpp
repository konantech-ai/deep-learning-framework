#include "../utils/tp_grid_canvas.h"
#include "../objects/tp_tensor.h"

GridCanvas::GridCanvas(string name, int rows, int cols, int hsize, int csize, int hgap, int cgap) {
    m_name = name;

    m_rows = rows;
    m_cols = cols;
    m_hsize = hsize;
    m_csize = csize;
    m_hgap = hgap;
    m_cgap = cgap;

    int height = m_rows * (m_hsize + m_hgap) + m_hgap;
    int width = m_cols * (m_csize + m_cgap) + m_cgap;
    int chns = 3;

    m_mat = TpUtils::createCanvas(height, width, chns);
}

GridCanvas::~GridCanvas() {
}

int GridCanvas::rows() {
    return m_rows;
}

int GridCanvas::cols() {
    return m_cols;
}

void GridCanvas::extendRow() {
    m_rows++;

    int height = m_rows * (m_hsize + m_hgap) + m_hgap;

    m_mat.resize(height, 0);
}

void GridCanvas::drawTensorImage(ETensor tensor, int ndat, int nrow, int ncol) {
    int x = ncol * (m_csize + m_cgap) + m_cgap;
    int y = nrow * (m_hsize + m_hgap) + m_hgap;

    cv::Rect rect(x, y, m_csize, m_hsize);

    TpUtils::drawTensorImage(m_mat, tensor, rect, ndat);
}

void GridCanvas::drawMapImage(ETensor tensor, int ndat, int nchn, int nrow, int ncol) {
    int x = ncol * (m_csize + m_cgap) + m_cgap;
    int y = nrow * (m_hsize + m_hgap) + m_hgap;

    cv::Rect rect(x, y, m_csize, m_hsize);

    TpUtils::addMapImage(m_mat, tensor, rect, ndat, nchn, 0x7);
}

void GridCanvas::drawMergedMapImage(ETensor tensor, int ndat, VIntList chns, int nrow, int ncol) {
    int x = ncol * (m_csize + m_cgap) + m_cgap;
    int y = nrow * (m_hsize + m_hgap) + m_hgap;

    cv::Rect rect(x, y, m_csize, m_hsize);

    TpUtils::addMapImage(m_mat, tensor, rect, ndat, chns[0], 0x1);
    TpUtils::addMapImage(m_mat, tensor, rect, ndat, chns[1], 0x2);
    TpUtils::addMapImage(m_mat, tensor, rect, ndat, chns[2], 0x4);
}

void GridCanvas::drawFilterImage(ETensor tensor, int xchn, int ychn, int nrow, int ncol, float coef) {
    int x = ncol * (m_csize + m_cgap) + m_cgap;
    int y = nrow * (m_hsize + m_hgap) + m_hgap;

    cv::Rect rect(x, y, m_csize, m_hsize);

    TpUtils::addFilterImage(m_mat, tensor, rect, xchn, ychn, 0x7, coef);
}

void GridCanvas::drawMergedFilterImage(ETensor tensor, VIntList xchns, VIntList ychns, int nrow, int ncol, float coef) {
    int x = ncol * (m_csize + m_cgap) + m_cgap;
    int y = nrow * (m_hsize + m_hgap) + m_hgap;

    cv::Rect rect(x, y, m_csize, m_hsize);

    TpUtils::addFilterImage(m_mat, tensor, rect, xchns[0], ychns[0], 0x1, coef);
    TpUtils::addFilterImage(m_mat, tensor, rect, xchns[1], ychns[1], 0x2, coef);
    TpUtils::addFilterImage(m_mat, tensor, rect, xchns[2], ychns[2], 0x4, coef);
}

void GridCanvas::displayImage(int nMilliSec) {
    TpUtils::displayImage(m_name, m_mat, nMilliSec);
}

void GridCanvas::saveImage(string folder, string filename) {
    filename = m_name + "_" + filename;
    TpUtils::saveImage(folder, filename, m_mat);
}
