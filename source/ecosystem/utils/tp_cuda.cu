#include "../utils/tp_cuda.h"
#include "../utils/tp_exception.h"
#include "../objects/tp_tensor.h"
#include "../objects/tp_nn.h"

#include "opencv2/imgcodecs.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui.hpp"

#include <cfloat>

int ms_block_size = 512;

TpCuda::TpCuda() {
	m_usingCuda = false;

	int nAvailDevices;
	cudaGetDeviceCount(&nAvailDevices);
	if (nAvailDevices > 0) {
		cudaGetDevice(&m_nOldDev);
		if (cudaSetDevice(1) == 0) { //nAvailDevices-1) == 0) {
			m_usingCuda = true;
		}
	}
}

TpCuda::~TpCuda() {
	if (m_usingCuda) {
		cudaSetDevice(m_nOldDev);
	}
}

TpCuda::CudaMemPiece::CudaMemPiece(ETensor x, bool input, bool output) {
	m_input = input;
	m_output = output;
	m_pDevMem = NULL;
	m_pHostMem = x.void_ptr();
	m_memSize = x.byteSize();

	if (cudaMalloc(&m_pDevMem, m_memSize) != 0) TP_THROW(VERR_TENSOR_DEVICE);
	if (input) cudaMemcpy(m_pDevMem, m_pHostMem, m_memSize, cudaMemcpyHostToDevice);
}

TpCuda::CudaMemPiece::CudaMemPiece(void* ptr, int64 size, bool input, bool output) {
	m_input = input;
	m_output = output;
	m_pDevMem = NULL;
	m_pHostMem = ptr;
	m_memSize = size;

	if (cudaMalloc(&m_pDevMem, m_memSize) != 0) TP_THROW(VERR_TENSOR_DEVICE);
	if (input) cudaMemcpy(m_pDevMem, m_pHostMem, m_memSize, cudaMemcpyHostToDevice);
}

TpCuda::CudaMemPiece::~CudaMemPiece() {
	if (m_pDevMem) {
		if (m_output) cudaMemcpy(m_pHostMem, m_pDevMem, m_memSize, cudaMemcpyDeviceToHost);
		cudaFree(m_pDevMem);
	}
}

TpCuda::CudaMemPiece::operator float* () const {
	return (float*)m_pDevMem;
}

TpCuda::CudaMemPiece::operator int* () const {
	return (int*)m_pDevMem;
}

TpCuda::CudaMemPiece::operator unsigned char* () const {
	return (uchar*)m_pDevMem;
}

__global__ void cuda_resize(int64 size, float* py, float* px, int64 ndat, int64 nchn, int64 xht, int64 xwd, int64 yht, int64 ywd) {
	int64 n = blockIdx.x * blockDim.x + threadIdx.x;

	if (n < size) {
		int64 yd = n / (nchn * yht * ywd);
		int64 yn = n / (yht * ywd) % nchn;
		int64 yr = n / ywd % yht;
		int64 yc = n % ywd;

		int64 xr = yr * xht / yht;
		int64 xc = yc * xwd / ywd;

		int64 xpos = (((yd * nchn) + yn) * xht + xr) * xwd + xc;

		py[n] = px[xpos];
	}
}

ETensor TpCuda::resize(ETensor x, VShape shape) {
	if (!x.isValid()) TP_THROW(VERR_INVALID_CORE);
	if (x.type() != VDataType::float32) TP_THROW(VERR_INVALID_CORE);

	VShape xshape = x.shape();

	int64 ndat = xshape[0], nchn = xshape[1], xht = xshape[2], xwd = xshape[3];
	int64 yht = shape[0], ywd = shape[1];

	VShape yshape{ ndat, nchn, yht, ywd };

	ETensor y(x.nn(), yshape, x.type());

	resize_on(y, x);

	return y;
}

__global__ void cuda_transpose(int64 size, float* py, float* px, int* pn, int64 axis_size, int64 data_size) {
	int64 n = blockIdx.x * blockDim.x + threadIdx.x;

	if (n < size) {
		int xpos = 0;
		int ypos = n;

		int cood;

		for (int64 m = 0; m < axis_size; m++) {
			int y_block_size = pn[m * 3 + 1];
			int x_block_size = pn[pn[m * 3] * 3 + 2];

			cood = ypos / y_block_size;
			ypos = ypos % y_block_size;

			xpos += cood * x_block_size;
		}

		py[n] = px[xpos];
	}
}

ETensor TpCuda::transpose(ENN nn, ETensor x, VList axes) {
	if (!x.isValid()) TP_THROW(VERR_INVALID_CORE);
	if (x.type() != VDataType::float32) TP_THROW(VERR_INVALID_CORE);

	VShape xshape = x.shape();
	VShape tshape;

	int64 axis_size = xshape.size();
	int64 data_size = xshape.total_size();

	if (axis_size != axes.size()) TP_THROW2(VERR_BAD_SHAPE_TENSOR, __func__);

	int mask1 = 0;
	int mask2 = 0;

	int64 rest = data_size;
	int64 prod = data_size;

	ETensor axinfo(nn, VShape{ axis_size, 3 }, VDataType::int32);

	for (int n = 0; n < axis_size; n++) {
		int axis = (int)axes[n];

		if (axis < 0 || axis >= axis_size) TP_THROW(VERR_OUT_OF_RANGE);

		tshape = tshape.append(xshape[axis]);

		mask1 |= (1 << n);
		mask2 |= (1 << axis);

		rest /= (int64)xshape[axis];
		prod /= (int64)xshape[n];

		axinfo.setElement(VList({ n, 0 }), axis);
		axinfo.setElement(VList({ n, 1 }), rest);
		axinfo.setElement(VList({ n, 2 }), prod);
	}

	if (mask1 != mask2) TP_THROW(VERR_TENSOR_MASK);

	ETensor y(nn, tshape, x.type());

	if (m_usingCuda) {
		CudaMemPiece cx(x, true, false);
		CudaMemPiece cy(y, false, true);
		CudaMemPiece cn(axinfo, true, false);

		int64 nSize = tshape.total_size();

		unsigned int nthreads = (unsigned int)((nSize + ms_block_size - 1) / ms_block_size);

		cuda_transpose << < nthreads, ms_block_size >> > (nSize, cy, cx, cn, axis_size, data_size);
	}
	else {
		TP_THROW(VERR_NOT_IMPLEMENTED_YET);
	}

	return y;
}

void TpCuda::resize_on(ETensor  y, ETensor x) {
	if (!x.isValid()) TP_THROW(VERR_UNDEFINED);
	if (!y.isValid()) TP_THROW(VERR_UNDEFINED);

	if (x.type() != VDataType::float32) TP_THROW(VERR_UNDEFINED);
	if (y.type() != VDataType::float32) TP_THROW(VERR_UNDEFINED);

	if (x.device() != -1) TP_THROW(VERR_UNDEFINED);
	if (y.device() != -1) TP_THROW(VERR_UNDEFINED);

	VShape xshape = x.shape();
	VShape yshape = y.shape();

	int64 ndat = xshape[0], nchn = xshape[1], xht = xshape[2], xwd = xshape[3];
	int64 ydat = yshape[0], ychn = yshape[1], yht = yshape[2], ywd = yshape[3];

	if (ndat != ydat) TP_THROW(VERR_UNDEFINED);
	if (nchn != ychn) TP_THROW(VERR_UNDEFINED);

	int64 nSize = yshape.total_size();

	if (m_usingCuda) {
		CudaMemPiece cx(x, true, false);
		CudaMemPiece cy(y, false, true);

		int64 nSize = yshape.total_size();

		unsigned int nthreads = (unsigned int)((nSize + ms_block_size - 1) / ms_block_size);

		cuda_resize << < nthreads, ms_block_size >> > (nSize, cy, cx, ndat, nchn, xht, xwd, yht, ywd);
	}
	else {
		float* px = x.float_ptr();
		float* py = y.float_ptr();

		int64 ypos = 0;

		for (int64 yd = 0; yd < ndat; yd++) {
			for (int64 yn = 0; yn < nchn; yn++) {
				for (int64 yr = 0; yr < yht; yr++) {
					int64 xr = yr * xht / yht;
					for (int64 yc = 0; yc < ywd; yc++) {
						int64 xc = yc * xwd / ywd;
						int64 xpos = (((yd * nchn) + yn) * xht + xr) * xwd + xc;
						py[ypos++] = px[xpos];
					}
				}
			}
		}
	}
}

__global__ void cuda_transpose_on(int64 size, float* py, float* px, int64 n1, int64 n2, int64 n3, int64 n4, int64 n5) {
	int64 n = blockIdx.x * blockDim.x + threadIdx.x;

	if (n < size) {
		int64 k1 = n / (n2 * n3 * n4 * n5);
		int64 k2 = n / (n3 * n4 * n5) % n2;
		int64 k3 = n / (n4 * n5) % n3;
		int64 k4 = n / n5 % n4;
		int64 k5 = n % n5;

		int64 xpos = ((((k1 * n4) + k4) * n3 + k3) * n2 + k2) * n5 + k5;

		py[n] = px[xpos];
	}
}

void TpCuda::transpose_on(ETensor x, int64 axis1, int64 axis2) {
	VShape xshape = x.shape();
	
	if (axis1 < 0 || axis1 >= xshape.size()) TP_THROW(VERR_UNDEFINED);
	if (axis2 < 0 || axis2 >= xshape.size()) TP_THROW(VERR_UNDEFINED);
	if (axis1 == axis2) TP_THROW(VERR_UNDEFINED);
	if (axis1 > axis2) {
		int64 temp = axis1;
		axis1 = axis2;
		axis2 = temp;
	}

	VShape tshape = xshape.copy();
	
	tshape[axis1] = xshape[axis2];
	tshape[axis2] = xshape[axis1];

	int64 n1 = tshape.head_size(axis1);
	int64 n2 = tshape[axis1];
	int64 n3 = tshape.head_size(axis2) / tshape.head_size(axis1 + 1);
	int64 n4 = tshape[axis2];
	int64 n5 = tshape.total_size() / tshape.head_size(axis2+1);

	if (m_usingCuda) {
		CudaMemPiece cx(x, true, false);
		CudaMemPiece cy(x, false, true);

		int64 nSize = tshape.total_size();

		unsigned int nthreads = (unsigned int)((nSize + ms_block_size - 1) / ms_block_size);

		cuda_transpose_on << < nthreads, ms_block_size >> > (nSize, cy, cx, n1, n2, n3, n4, n5);
	}
	else {
		TP_THROW(VERR_UNDEFINED);
	}
}

__global__ void cuda_resize_pixels(int64 size, float* py, unsigned char* px, int64 nrow, int64 ncol, int64 nchn, int64 height, int64 width, bool chn_last, bool transpose, float mix) {
	int64 n = blockIdx.x * blockDim.x + threadIdx.x;

	if (n < size) {
		int64 nr = n / (ncol * nchn) % nrow;
		int64 nc = n / nchn % ncol;
		int64 nn = n % nchn;

		float left = nc * width / (float)ncol;
		float right = (nc + 1) * width / (float)ncol;
		float top = nr * height / (float)nrow;
		float bottom = (nr + 1) * height / (float)nrow;

		float xratio = (float)ncol / width;		// 입력 한 픽셀의 너비가 출력 한 픽셀 너비에 반영되어야 하는 비율
		float yratio = (float)nrow / height;	// 입력 한 픽셀의 높이가 출력 한 픽셀 높이에 반영되어야 하는 비율

		float ypixel = 0;

		for (int64 nx = (int64)left; (float)nx < right; nx++) {
			float xt = MIN(nx + 1, right) - MAX(nx, left);		// nx 위치 입력 픽셀의 너비 1 중에서 nc 출력 픽셀에 속하는 비율
			for (int64 ny = (int64)top; (float)ny < bottom; ny++) {
				float yt = MIN(ny + 1, bottom) - MAX(ny, top);	// ny 위치 입력 픽셀의 높이 1 중에서 nr 출력 픽셀에 속하는 비율
				int64 xpos = (ny * width + nx) * nchn + nn;
				float xpixel = px[xpos] / 255.0f;
				ypixel += xpixel * (xt * xratio) * (yt * yratio);
			}
		}

		int64 ypos;
		
		if (chn_last) {
			ypos = transpose ? (nc * nrow + nr) * nchn + nn : (nr * ncol + nc) * nchn + nn;
		}
		else {
			ypos = transpose ? (nn * ncol + nc) * nrow + nr : (nn * nrow + nr) * ncol + nc;
		}

		py[ypos] = ypixel;
		//py[ypos] = mix* ypixel + (1 - mix) * py[ypos];
	}
}

void TpCuda::load_jpeg_pixels(ETensor x, string filepath, bool chn_last, bool transpose, int code, float mix) {
#ifdef KA_WINDOWS
	std::replace(filepath.begin(), filepath.end(), '/', '\\');
#else
	std::replace(filepath.begin(), filepath.end(), '\\', '/');
#endif

	cv::Mat mat = cv::imread(filepath, cv::IMREAD_COLOR);
	cv::Mat img;

	if (code >= 0) cv::cvtColor(mat, img, code);
	else img = mat;

	VShape xshape = x.shape();

	if (xshape.size() != 3 || mat.dims != 2) TP_THROW(VERR_UNDEFINED);

	int64 nrow = xshape[chn_last ? 0 : 1];
	int64 ncol = xshape[chn_last ? 1 : 2];
	int64 nchn = xshape[chn_last ? 2 : 0];

	int64 img_size = img.rows * img.cols * img.channels();

	int height = img.rows;
	int width = img.cols;

	if (m_usingCuda) {
		CudaMemPiece ci(img.data, img_size, true, false);
		CudaMemPiece cx(x, false, true);

		int64 nSize = xshape.total_size();

		unsigned int nthreads = (unsigned int)((nSize + ms_block_size - 1) / ms_block_size);

		cuda_resize_pixels << < nthreads, ms_block_size >> > (nSize, cx, ci, nrow, ncol, nchn, height, width, chn_last, transpose, mix);
	}
	else {
		TP_THROW(VERR_UNDEFINED);
	}
}

__global__ void m_copy(int64 size, float* py, float* px) {
	int64 n = blockIdx.x * blockDim.x + threadIdx.x;

	if (n < size) {
		py[n] = px[n];
	}
}

void TpCuda::copy(float* py, float* px, int64 nCount) {
	int nOldDev;
	
	cudaGetDevice(&nOldDev);
	cudaSetDevice(0);

	float* pcy;
	float* pcx;
	
	cudaMalloc(&pcy, sizeof(float) * nCount);
	cudaMalloc(&pcx, sizeof(float) * nCount);

	cudaMemcpy(pcx, px, sizeof(float) * nCount, cudaMemcpyHostToDevice);

	unsigned int nthreads = (unsigned int)((nCount + ms_block_size - 1) / ms_block_size);

	m_copy << < nthreads, ms_block_size >> > (nCount, pcy, pcx);

	cudaMemcpy(py, pcy, sizeof(float) * nCount, cudaMemcpyDeviceToHost);

	cudaFree(pcx);
	cudaFree(pcy);

	cudaSetDevice(nOldDev);

	cudaError_t cuda_ret = cudaGetLastError();

	if (cuda_ret != 0) {
		string sCudaError = cudaGetErrorString(cuda_ret);
		print("[TpCuda Error] %s in %s:%d", sCudaError.c_str(), __FILE__, __LINE__);
		TP_THROW(VERR_CUDA_ERROR);
	}
}

__global__ void m_max_forward(int64 size, float* py, float* px, int* pm, int* ps, int64 xdat, int64 xrow, int64 xcol, int64 xchn, int64 krow, int64 kcol, float ratio1, float ratio2) {
	int64 n = blockIdx.x * blockDim.x + threadIdx.x;

	if (n < size) {
		int64 nd = n / (xrow * xcol * xchn);
		int64 nr = n / (xcol * xchn) % xrow;
		int64 nc = n / xchn % xcol;
		int64 nn = n % xchn;

		int64 br = (krow - 1) / 2;
		int64 bc = (kcol - 1) / 2;

		float max1 = -FLT_MAX;
		float max2 = -FLT_MAX;

		int64 m_idx1 = -1;
		int64 m_idx2 = -1;

		for (int64 kr = 0; kr < krow; kr++) {
			for (int64 kc = 0; kc < kcol; kc++) {
				int64 xr = nr + kr - br;
				int64 xc = nc + kc - bc;

				if (xr < 0 || xr >= xrow) continue;
				if (xc < 0 || xc >= xcol) continue;

				int64 xpos = ((nd * xrow + xr) * xcol + xc) * xchn + nn;
				float x = px[xpos];
				if (x > max1) {
					max2 = max1;
					m_idx2 = m_idx1;

					max1 = x;
					m_idx1 = kr * kcol + kc;
				}
				else if (x > max2) {
					max2 = x;
					m_idx2 = kr * kcol + kc;
				}
			}
		}

		py[n] = max1 * ratio1 + max2 * ratio2;

		pm[n] = (int)m_idx1;
		ps[n] = (int)m_idx2;
	}
}

void TpCuda::max_forward(float* py, float* px, int* pm, int* ps, int64 xdat, int64 xrow, int64 xcol, int64 xchn, int64 krow, int64 kcol, bool useKernel, bool use2nd) {
	if (!useKernel) {
		krow = 2;
		kcol = 2;
	}

	float ratio_1st = use2nd ? 0.7f : 1.0f;
	float ratio_2nd = use2nd ? 0.3f : 0.0f;

	int nOldDev;

	cudaGetDevice(&nOldDev);
	cudaSetDevice(0);

	float* pcy;
	float* pcx;

	int* pcm;
	int* pcs;

	int64 nSize = xdat * xrow * xcol * xchn;

	cudaMalloc(&pcy, sizeof(float) * nSize);
	cudaMalloc(&pcx, sizeof(float) * nSize);
	cudaMalloc(&pcm, sizeof(int) * nSize);
	cudaMalloc(&pcs, sizeof(int) * nSize);

	cudaMemcpy(pcx, px, sizeof(float) * nSize, cudaMemcpyHostToDevice);

	unsigned int nthreads = (unsigned int)((nSize + ms_block_size - 1) / ms_block_size);

	m_max_forward << < nthreads, ms_block_size >> > (nSize, pcy, pcx, pcm, pcs, xdat, xrow, xcol, xchn, kcol, krow, ratio_1st, ratio_2nd);

	cudaMemcpy(py, pcy, sizeof(float) * nSize, cudaMemcpyDeviceToHost);
	cudaMemcpy(pm, pcm, sizeof(int) * nSize, cudaMemcpyDeviceToHost);
	cudaMemcpy(pm, pcs, sizeof(int) * nSize, cudaMemcpyDeviceToHost);

	cudaFree(pcx);
	cudaFree(pcy);
	cudaFree(pcm);
	cudaFree(pcs);

	cudaSetDevice(nOldDev);

	cudaError_t cuda_ret = cudaGetLastError();

	if (cuda_ret != 0) {
		string sCudaError = cudaGetErrorString(cuda_ret);
		print("[TpCuda Error] %s in %s:%d", sCudaError.c_str(), __FILE__, __LINE__);
		TP_THROW(VERR_CUDA_ERROR);
	}
}

__global__ void m_max_backward(int64 size, float* py, float* px, int* pm, int* ps, int64 xdat, int64 xrow, int64 xcol, int64 xchn, int64 krow, int64 kcol, float ratio1, float ratio2) {
	int64 n = blockIdx.x * blockDim.x + threadIdx.x;

	if (n < size) {
		int64 nd = n / (xrow * xcol * xchn);
		int64 nr = n / (xcol * xchn) % xrow;
		int64 nc = n / xchn % xcol;
		int64 nn = n % xchn;

		int64 br = (krow - 1) / 2;
		int64 bc = (kcol - 1) / 2;

		float sum = 0;

		for (int64 kr = 0; kr < krow; kr++) {
			for (int64 kc = 0; kc < kcol; kc++) {
				int64 yr = nr - kr + br;
				int64 yc = nc - kc + bc;

				if (yr < 0 || yr >= xrow) continue;
				if (yc < 0 || yc >= xcol) continue;

				int64 ypos = ((nd * xrow + yr) * xcol + yc) * xchn + nn;
				float gy = py[ypos];
				
				int64 m_idx1 = pm[ypos];
				int64 m_idx2 = ps[ypos];

				if (m_idx1 == kr * kcol + kc) {
					sum += gy * ratio1;
				}
				else if (m_idx2 == kr * kcol + kc) {
					sum += gy * ratio2;
				}
			}
		}

		px[n] = sum;
	}
}

void TpCuda::max_backward(float* pgx, float* pgy, int* pm, int* ps, int64 xdat, int64 xrow, int64 xcol, int64 xchn, int64 krow, int64 kcol, bool useKernel, bool use2nd) {
	if (!useKernel) {
		krow = 2;
		kcol = 2;
	}

	float ratio_1st = use2nd ? 0.7f : 1.0f;
	float ratio_2nd = use2nd ? 0.3f : 0.0f;

	int nOldDev;

	cudaGetDevice(&nOldDev);
	cudaSetDevice(0);

	float* pcy;
	float* pcx;

	int* pcm;
	int* pcs;

	int64 nSize = xdat * xrow * xcol * xchn;

	cudaMalloc(&pcx, sizeof(float) * nSize);
	cudaMalloc(&pcy, sizeof(float) * nSize);
	cudaMalloc(&pcm, sizeof(int) * nSize);
	cudaMalloc(&pcs, sizeof(int) * nSize);

	cudaError_t cuda_ret = cudaGetLastError();

	if (cuda_ret != 0) {
		string sCudaError = cudaGetErrorString(cuda_ret);
		print("[TpCuda Error] %s in %s:%d", sCudaError.c_str(), __FILE__, __LINE__);
		TP_THROW(VERR_CUDA_ERROR);
	}

	cudaMemcpy(pcy, pgy, sizeof(float) * nSize, cudaMemcpyHostToDevice);
	cudaMemcpy(pcm, pm, sizeof(int) * nSize, cudaMemcpyHostToDevice);
	cudaMemcpy(pcs, ps, sizeof(int) * nSize, cudaMemcpyHostToDevice);

	unsigned int nthreads = (unsigned int)((nSize + ms_block_size - 1) / ms_block_size);

	m_max_backward << < nthreads, ms_block_size >> > (nSize, pcy, pcx, pcm, pcs, xdat, xrow, xcol, xchn, kcol, krow, ratio_1st, ratio_2nd);

	cudaMemcpy(pgx, pcx, sizeof(float) * nSize, cudaMemcpyDeviceToHost);

	cudaFree(pcx);
	cudaFree(pcy);
	cudaFree(pcm);
	cudaFree(pcs);

	cudaSetDevice(nOldDev);

	cuda_ret = cudaGetLastError();

	if (cuda_ret != 0) {
		string sCudaError = cudaGetErrorString(cuda_ret);
		print("[TpCuda Error] %s in %s:%d", sCudaError.c_str(), __FILE__, __LINE__);
		TP_THROW(VERR_CUDA_ERROR);
	}
}
