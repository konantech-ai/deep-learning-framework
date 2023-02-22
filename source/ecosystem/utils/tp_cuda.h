#pragma once

#include <cuda_runtime.h>
#include <curand.h>
#include <curand_kernel.h>

#include "../utils/tp_common.h"
#include "../objects/tp_tensor.h"

class TpCuda {
public:
	TpCuda();
	virtual ~TpCuda();

public:
	// new style usage in trying
	ETensor resize(ETensor x, VShape shape);
	ETensor transpose(ENN nn, ETensor x, VList axes);

	void resize_on(ETensor y, ETensor x);
	void transpose_on(ETensor x, int64 axis1, int64 axis2);
	void load_jpeg_pixels(ETensor x, string filepath, bool chn_last, bool transpose, int code, float mix);

protected:
	bool m_usingCuda;

	int m_nOldDev;

public:
	// old style usage
	static void copy(float* py, float* px, int64 nCount);

	static void max_forward(float* py, float* px, int* pm, int* ps, int64 xdat, int64 xrow, int64 xcol, int64 xchn, int64 krow, int64 kcol, bool useKernel, bool use2nd);
	static void max_backward(float* pgx, float* pgy, int* pm, int* ps, int64 xdat, int64 xrow, int64 xcol, int64 xchn, int64 krow, int64 kcol, bool useKernel, bool use2nd);

protected:
	class CudaMemPiece {
	public:
		CudaMemPiece(ETensor x, bool input, bool output);
		CudaMemPiece(void* ptr, int64 size, bool input, bool output);
		~CudaMemPiece();

		operator float* () const;
		operator int* () const;
		operator unsigned char* () const;

	protected:
		bool m_input;
		bool m_output;
		void* m_pDevMem;
		void* m_pHostMem;
		int64 m_memSize;
	};
};
