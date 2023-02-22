/*K
This file is part of Konantech AI Framework project
It is subject to the license terms in the LICENSE file found in the top-level directory of this distribution
*/
// kpy_data_handler.h
#pragma once

#include <vector>
#include <stdio.h>
#include <assert.h>
#include <sstream>
#include <iostream>

#ifdef KPY_EXPORTS
	#ifdef KPY_API
		#undef KPY_API
	#endif
	#define KPY_API __declspec(dllexport)
#else
	#define KPY_API __declspec(dllimport)
#endif

#ifdef FOR_LINUX
	#ifdef KPY_API
		#undef KPY_API
	#endif
	#define KPY_API __attribute__((__visibility__("default")))
#endif

// For python API
#ifdef KPY_EXPORTS
	#define PY_SSIZE_T_CLEAN
	#include <Python.h>
#else
	// A python object.
	class PyObject;
#endif


////////////////////////////////////////////////////////////////////////////////////////////////////
// Forward declarations
////////////////////////////////////////////////////////////////////////////////////////////////////

class VValue;	// Value data structure used in KonanTech AI engine. It can contain various types of data.
class VDict;	// Dictionary data structure used in KonanTech AI engine. It is compatible with python.
class VList;	// List data structure used in KonanTech AI engine. It is compatible with python.
class VShape;	// Shape data structure used in KonanTech AI engine. It is compatible with python's tuple.

namespace kpy {
	////////////////////////////////////////////////////////////////////////////////////////////////////
	// C++ methods for conversion from Python to KAI engine
	////////////////////////////////////////////////////////////////////////////////////////////////////
	
	/**
	* Convert the data from PyObject in python to VValue used in KonanTech AI engine.
	* @param [in]  src  If non-null, a pointer of the python object.
	* @param [out] dst  If non-null, a pointer of the VValue object.
	*/
	void ConvertPyObjectToVValue(PyObject* src, VValue* dst);

	/**
	* Convert the data from a dictionary in python to a VDict used in KonanTech AI engine.
	* (Caution! In python, (N) is evaluated as an int, and (N,) is evaluated as a tuple.)
	* @param [in]  src  If non-null, a pointer of the PyDict object.
	* @param [out] dst  If non-null, a pointer of the VDict object.
	*/
	void ConvertPyDictToVDict(PyObject* src, VDict* dst);

	/**
	* Convert the data from a list in python to a VList used in KonanTech AI engine.
	* @param [in]  src  If non-null, a pointer of the PyList object.
	* @param [out] dst  If non-null, a pointer of the VList object.
	*/
	void ConvertPyListToVList(PyObject* src, VList* dst);

	/**
	* Convert the data from a slice or int index in python to a VList used in KonanTech AI engine.
	* @param [in]  src  If non-null, a pointer of the PySlice object.
	* @param [in]  shape destination tensor shape to check the indexing values
	* @param [out] dst  If non-null, a pointer of the VList object for slice VList[strart, stop, step] and for int index index itself
	* @param [in] size  if positive must be equal the total size of element space
	*/
	void ConvertPySliceIndexToVList(PyObject* src, VShape shape, VList* dst, int64 nSize=0);

	/**
	* Convert the data from a list in python to a vector in C++.
	* @param [in]  src  If non-null, a pointer of the PyList object.
	* @param [out] dst  If non-null, a pointer of the vector object in C++.
	*/
	void ConvertPyListToVector(PyObject* src, std::vector<void*>* dst);

	/**
	* Convert the data from a list in python to a int64 array in C++.
	* @param [in]  src  If non-null, a pointer of the PyList object.
	* @param [out] pnSize  If non-null, a pointer of the allocated size
	* @returns NULL if it fails, else a pointer to an allocated int array, need to free using delete[] command outside
	*/
	int64* ConvertPyListToIntArr(PyObject* src, int64* pnSize);

	/**
	* Convert the data from a tuple in python to a VList used in KonanTech AI engine.
	* @param [in]  src  If non-null, a pointer of the PyTuple object.
	* @param [out] dst  If non-null, a pointer of the VList object.
	*/
	void ConvertPyTupleToVList(PyObject* src, VList* dst);

    /**
	* Convert the data from a tuple in python to a VShape used in KonanTech AI engine.
	* @param [in]  src  If non-null, a pointer of the PyTuple object.
	* @param [out] dst  If non-null, a pointer of the VShape object.
	*/
	void ConvertPyTupleToVShape(PyObject* src, VShape* dst);



	////////////////////////////////////////////////////////////////////////////////////////////////////
	// C++ methods for conversion from KAI engine to Python
	////////////////////////////////////////////////////////////////////////////////////////////////////

	/**
	* Convert the data from a VValue in KaiEngine to a term in python.
	* @param   VValue in Kai Engine
	* @returns NULL if it fails, else a pointer to data in python.
	*/
	PyObject* ConvertVValueToPyObject(VValue src);
	PyObject* m_convertVValueToPyObject(VValue src);

	/**
	* Convert the data from a VDict in KaiEngine to a dict in python.
	* @param   VDict in Kai Engine
	* @returns NULL if it fails, else a pointer to dict data in python.
	*/
	PyObject* ConvertVDictToPyDict(VDict src);
	PyObject* m_convertVDictToPyDict(VDict src);

	/**
	* Convert the data from a VList in KaiEngine to a list in python.
	* @param   VList in Kai Engine
	* @returns NULL if it fails, else a pointer to list data in python.
	*/
	PyObject* ConvertVListToPyList(VList src);
	PyObject* m_convertVListToPyList(VList src);
	
	PyObject* ConvertVStrListToPyList(VStrList src);

	/**
	* Convert the data from a VShape in KaiEngine to a list in python.
	* @param   VShape in Kai Engine
	* @returns NULL if it fails, else a pointer to list data in python.
	*/
	PyObject* ConvertVShapeToPyList(VShape src);
	PyObject* m_convertVShapeToPyList(VShape src);

	/**
	* Convert the data from a vector in C++ to a list in python.
	* @param   vector data in C++
	* @returns NULL if it fails, else a pointer to list data in python.
	*/
	PyObject* ConvertVectorToPyList(const std::vector<double>& src);

	/**
	* Convert the data from a vector in C++ to a list in python.
	* @param   vector data in C++
	* @returns NULL if it fails, else a pointer to list data in python.
	*/
	PyObject* ConvertVectorToPyList(const std::vector<void*>& src);

	/**
	* Convert the data from a vector in C++ to a tuple in python.
	* @param   vector data in C++
	* @returns NULL if it fails, else a pointer to tuple data in python.
	*/
	PyObject* ConvertVectorToPyTuple(const std::vector<double>& src);

	/**
	* Convert the data from a vector in C++ to a tuple in python.
	* @param   vector data in C++
	* @returns NULL if it fails, else a pointer to tuple data in python.
	*/
	PyObject* ConvertVectorToPyTuple(const std::vector<void*>& src);

	/**
	* Convert the data from a 2-D vector in C++ to a tuple in python.
	* @param   vector data in C++
	* @returns NULL if it fails, else a pointer to tuple data in python.
	*/
	PyObject* ConvertVector2DToPyTuple(const std::vector< std::vector<double> >& src);

	/**
	* Convert 2-D Array from float32 pointer in C++ to a list in python.
	* @param   float32 pointer, 2-D Array Shape row, 2-D Array Shape col in C++
	* @returns NULL if it fails, else a pointer to list data in python.
	*/
	PyObject* ConvertArray2Dpf32ToPyList(void* Array2Dpf32, int Array2Drow, int Array2Dcol);

	/**
	* Convert the data from a VShape used in KonanTech AI engine to a tuple in python.
	* @param [in]  src  If non-null, a pointer of the VShape object
    * @returns NULL if it fails, else a pointer to tuple in python.
	*/
	PyObject* ConvertVShapeToPyTuple(const VShape* src);

};


#ifdef __cplusplus
extern "C" {
#endif
	////////////////////////////////////////////////////////////////////////////////////////////////////
	// Python methods for handling python objects
	////////////////////////////////////////////////////////////////////////////////////////////////////

	// Print out a type name of PyObject variable.
	KPY_API void  PrintPyObjectType(PyObject* py_object);

	// Print out 1-D numpy array.
	KPY_API void  PrintNumpy1DAsDouble(double* numpy_obj, int array_size);

	// Print out data of the list or tuple type.
	KPY_API void  PrintListAndTupleAsDouble(PyObject* py_object);

#ifdef __cplusplus
}
#endif
