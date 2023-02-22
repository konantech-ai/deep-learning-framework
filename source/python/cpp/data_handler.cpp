/*K
This file is part of Konantech AI Framework project
It is subject to the license terms in the LICENSE file found in the top-level directory of this distribution
*/

// Reference : Interconversion between C++ vectors and Python list, tuple (https://gist.github.com/rjzak/5681680)
// Reference : Python Dictionary (https://docs.python.org/ko/3.10/c-api/dict.html)

//#include <include/vapi.h>
#include <include/vvalue.h>

#include "data_handler.h"

////////////////////////////////////////////////////////////////////////////////////////////////////
// Type definitions
////////////////////////////////////////////////////////////////////////////////////////////////////

typedef Py_ssize_t (*CFUNCTYPE_PyListAndTuple_Size)(PyObject*);
typedef PyObject*  (*CFUNCTYPE_PyListAndTuple_GetItem)(PyObject*, Py_ssize_t);


////////////////////////////////////////////////////////////////////////////////////////////////////
// C++ methods for conversion between and Python and KAI engine
////////////////////////////////////////////////////////////////////////////////////////////////////

namespace kpy {
	/* C++ methods for conversion from Python to KAI engine */

	void ConvertPyObjectToVValue(PyObject* src, VValue* dst) {
		if (dst == NULL) {
			throw std::logic_error("VValue* argument is invalid");
		}

		if (src == Py_None) {
			*dst = VValue();
			return;
		}
		else if (PyBool_Check(src)) {
			if (src == Py_True) {
				bool val = true;
				*dst = val;
				return;
			}
			else if (src == Py_False) {
				bool val = false;
				*dst = val;
				return;
			}
			else {
				throw std::logic_error("PyObject* argument must be Py_True or Py_False");
			}
		}
		else if (PyFloat_Check(src)) {
			double val = PyFloat_AsDouble(src);
			*dst = val;
			return;
		}
		else if (PyLong_Check(src)) {
			long long val = PyLong_AsLongLong(src);
			*dst = (int64)val;
			return;
		}
		else if (PyList_Check(src)) {
			VList list;
			ConvertPyListToVList(src, &list);
			*dst = list;
			return;
		}
		else if (PyTuple_Check(src)) {
			VShape shape;
			ConvertPyTupleToVShape(src, &shape);
			*dst = shape;
			return;
		}
		else if (PyBytes_Check(src)) {
			string str = PyBytes_AsString(src);
			*dst = str;
			return;
		}
		else if (PyUnicode_Check(src)) {
			Py_ssize_t len = 0;
			string str = PyUnicode_AsUTF8AndSize(src, &len);
			*dst = str;
			return;
		}
		else if (PyDict_Check(src)) {
			VDict dict;
			ConvertPyDictToVDict(src, &dict);
			*dst = dict;
			return;
		}
		else {
			// To avoid the bug of Python 3.9.x
			string type_name = src->ob_type->tp_name;
			std::transform(type_name.cbegin(), type_name.cend(), type_name.begin(), ::tolower);

			if (type_name == "float") {
				*dst = ((PyFloatObject*)src)->ob_fval;
				return;
			}
			else if (type_name == "tensor") {
				*dst = src;
				return;
			}
			else if (PyType_Check(src)) {
				fprintf(stderr, "error: PyType_Check() returns true.\n");
				throw std::logic_error("PyType_Check(): not implemented yet");
			}
			else {
				fprintf(stderr, "error: PyType_Check() returns false.\n");
				std::string error_log = "unknown type (" + type_name + ")";
				throw std::logic_error(error_log.c_str());
			}
		}
	}

	void ConvertPyDictToVDict(PyObject* src, VDict* dst) {
		if (src == Py_None) return;

		if (src == NULL || !PyDict_Check(src)) {
			throw std::logic_error("PyObject* argument is invalid");
		}
		if (dst == NULL || dst->size() > 0) {
			throw std::logic_error("VDict* argument is invalid");
		}

		Py_ssize_t pos = 0;		// Iterator's index
		PyObject* py_key = NULL;
		PyObject* py_val = NULL;

		// Index(pos) based iteration
		while (PyDict_Next(src, &pos, &py_key, &py_val)) {
			if (!PyUnicode_Check(py_key))
				throw std::logic_error("dict key must be a string to convert Vdict");
			Py_ssize_t len = 0;
			std::string key = PyUnicode_AsUTF8AndSize(py_key, &len);
			VValue val;
			ConvertPyObjectToVValue(py_val, &val);
			(*dst)[key] = val;
		}
	}

	void ConvertPyListToVList(PyObject* src, VList* dst) {
		if (dst == NULL || dst->size() > 0) {
			throw std::logic_error("VList argument is invalid");
		}

		PyObject* pIncoming = (PyObject*)src;

		CFUNCTYPE_PyListAndTuple_Size    cbPyListAndTuple_Size = NULL;
		CFUNCTYPE_PyListAndTuple_GetItem cbPyListAndTuple_GetItem = NULL;

		if (PyList_Check(pIncoming)) {
			cbPyListAndTuple_Size = PyList_Size;
			cbPyListAndTuple_GetItem = PyList_GetItem;
		}
		else {
			throw std::logic_error("Passed PyObject pointer was not a list");
		}

		for (Py_ssize_t i = 0; i < cbPyListAndTuple_Size(pIncoming); ++i) {
			PyObject* py_val = cbPyListAndTuple_GetItem(pIncoming, i);
			VValue val;
			ConvertPyObjectToVValue(py_val, &val);
			dst->push_back(val);
		}
	}

	void ConvertPySliceIndexToVList(PyObject* src, VShape shape, VList* dst, int64 nSize) {
		try {
			if (dst == NULL || dst->size() > 0) {
				throw std::logic_error("VList argument is invalid");
			}

			if (PyLong_Check(src)) {
				long long index = PyLong_AsLongLong(src);
				if (shape.size() != 1) throw std::logic_error("slice indexing need for multi-dimension tensor");
				if (index < 0) index += shape[0];
				if (index < 0 || index >= shape[0]) throw std::logic_error("index out of range");
				*dst = VList{ index };
				if (nSize > 0 && nSize != shape.total_size() / shape[0]) throw std::logic_error("bad size value in int index");
				return;
			}

			if (!PyTuple_Check(src)) {
				throw std::logic_error("bad indeing type");
			}

			Py_ssize_t index_count = PyTuple_Size(src);

			if (index_count > shape.size()) throw std::logic_error("bad size of slice index");

			VList index_list;
			int64 size = 1;

			for (Py_ssize_t i = 0; i < index_count; ++i) {
				PyObject* py_element = PyTuple_GetItem(src, i);

				if (PyLong_Check(py_element)) {
					long long index = PyLong_AsLongLong(py_element);
					if (index < 0) index += shape[i];
					if (index < 0 || index >= shape[i]) throw std::logic_error("index out of range");
					index_list.push_back(index);
				}
				else if (PySlice_Check(py_element)) {
					Py_ssize_t start, stop, step, length;
					if (PySlice_GetIndicesEx(py_element, shape[i], &start, &stop, &step, &length) != 0) {
						throw std::logic_error("slice format error");
					}

					index_list.push_back(VList{ (int64)start, (int64)stop, (int64)step });

					size *= length;
				}
				else {
					throw std::logic_error("bad indeing type");
				}
			}

			*dst = index_list;

			if (index_count > shape.size()) {
				VShape tshape = shape.tail(index_count);
				size *= tshape.total_size();
			}

			if (nSize > 0 && nSize != size) {
				throw std::logic_error("bad size value in tuple slice");
			}
		}
		catch (const exception& e)
		{
			cerr << "Caught: " << e.what() << endl;
			cerr << "Type: " << typeid(e).name() << endl;
		}
	}

	void ConvertPyListToVector(PyObject* src, std::vector<void*>* dst) {
		if (dst == NULL || dst->size() > 0) {
			throw std::logic_error("VList argument is invalid");
		}

		PyObject* pIncoming = (PyObject*)src;

		CFUNCTYPE_PyListAndTuple_Size    cbPyListAndTuple_Size = NULL;
		CFUNCTYPE_PyListAndTuple_GetItem cbPyListAndTuple_GetItem = NULL;

		if (PyList_Check(pIncoming)) {
			cbPyListAndTuple_Size = PyList_Size;
			cbPyListAndTuple_GetItem = PyList_GetItem;
		}
		else {
			throw std::logic_error("Passed PyObject pointer was not a list");
		}

		for (Py_ssize_t i = 0; i < cbPyListAndTuple_Size(pIncoming); ++i) {
			PyObject* py_val = cbPyListAndTuple_GetItem(pIncoming, i);
			dst->push_back((void*)PyLong_AsLongLong(py_val));
		}
	}

	int64* ConvertPyListToIntArr(PyObject* src, int64* pnSize) {
		PyObject* pIncoming = (PyObject*)src;

		CFUNCTYPE_PyListAndTuple_Size    cbPyListAndTuple_Size = NULL;
		CFUNCTYPE_PyListAndTuple_GetItem cbPyListAndTuple_GetItem = NULL;

		if (PyList_Check(pIncoming)) {
			cbPyListAndTuple_Size = PyList_Size;
			cbPyListAndTuple_GetItem = PyList_GetItem;
		}
		else {
			throw std::logic_error("Passed PyObject pointer was not a list");
		}

		int64 list_size = (int64)cbPyListAndTuple_Size(pIncoming);
		if (pnSize) *pnSize = list_size;

		if (list_size == 0) {
			return NULL;
		}

		int64* pnArr = new int64[list_size];
		if (pnArr == NULL) {
			throw std::logic_error("Memory allocation failure");
		}

		for (Py_ssize_t i = 0; i < list_size; ++i) {
			PyObject* py_val = cbPyListAndTuple_GetItem(pIncoming, i);
			pnArr[i] = PyLong_AsLongLong(py_val);
		}

		return pnArr;
	}

	void ConvertPyTupleToVList(PyObject* src, VList* dst) {
		if (dst == NULL || dst->size() > 0) {
			throw std::logic_error("VShape argument is invalid");
		}

		CFUNCTYPE_PyListAndTuple_Size    cbPyListAndTuple_Size = NULL;
		CFUNCTYPE_PyListAndTuple_GetItem cbPyListAndTuple_GetItem = NULL;

		if (PyTuple_Check(src)) {
			cbPyListAndTuple_Size = PyTuple_Size;
			cbPyListAndTuple_GetItem = PyTuple_GetItem;
		}
		else if (PyLong_Check(src)) {
			long long val = PyLong_AsLongLong(src);
			*dst = VList{ val };
			return;
		}
		else {
			printf("here here\n");
			throw std::logic_error("Passed PyObject pointer was not a tuple");
		}

		for (Py_ssize_t i = 0; i < cbPyListAndTuple_Size(src); ++i) {
			PyObject* py_val = cbPyListAndTuple_GetItem(src, i);
			VValue val;
			ConvertPyObjectToVValue(py_val, &val);
			dst->push_back((int64)val);
		}
	}

	void ConvertPyTupleToVShape(PyObject* src, VShape* dst) {
		if (dst == NULL || dst->size() > 0) {
			throw std::logic_error("VShape argument is invalid");
		}

		PyObject* pIncoming = (PyObject*)src;

		CFUNCTYPE_PyListAndTuple_Size    cbPyListAndTuple_Size = NULL;
		CFUNCTYPE_PyListAndTuple_GetItem cbPyListAndTuple_GetItem = NULL;

		if (PyTuple_Check(pIncoming)) {
			cbPyListAndTuple_Size = PyTuple_Size;
			cbPyListAndTuple_GetItem = PyTuple_GetItem;
		}
		else {
			throw std::logic_error("Passed PyObject pointer was not a tuple");
		}

		for (Py_ssize_t i = 0; i < cbPyListAndTuple_Size(pIncoming); ++i) {
			PyObject* py_val = cbPyListAndTuple_GetItem(pIncoming, i);
			VValue val;
			ConvertPyObjectToVValue(py_val, &val);
			if (val.type() == VValueType::int64) {
				*dst = dst->append((int64)val);
			}
			else if (val.type() == VValueType::shape) {
				VShape temp = val;
				*dst = dst->append(temp);
			}
			else {
				throw std::logic_error("Converted VValue instance was not a int64 or VShape");
			}
		}
	}

	/* C++ methods for conversion from KAI engine to Python */

	PyObject* ConvertVValueToPyObject(VValue src) {
		PyGILState_STATE gil_state = PyGILState_Ensure();

		PyObject* pyobject_dict = m_convertVValueToPyObject(src);

		PyGILState_Release(gil_state);

		return pyobject_dict;
	}

	PyObject* m_convertVValueToPyObject(VValue src) {
		PyObject* value;
		switch (src.type()) {
		case VValueType::int32:
			value = PyLong_FromLong((int)src);
			break;
		case VValueType::int64:
			value = PyLong_FromLongLong((int64)src);
			break;
		case VValueType::float32:
			value = PyFloat_FromDouble((double)src);
			break;
		case VValueType::kbool:
			value = PyBool_FromLong((int)src);
			break;
		case VValueType::string:
			value = PyUnicode_FromString(((string)src).c_str());
			break;
		case VValueType::list:
			value = m_convertVListToPyList((VList)src);
			break;
		case VValueType::dict:
			value = m_convertVDictToPyDict((VDict)src);
			break;
		case VValueType::shape:
			value = m_convertVShapeToPyList((VShape)src);
			break;
		default:
			throw std::runtime_error("Not implemented yet");
		}
		return value;
	}

	PyObject* ConvertVDictToPyDict(VDict src) {
		PyGILState_STATE gil_state = PyGILState_Ensure();

		PyObject* pyobject_dict = m_convertVDictToPyDict(src);

		PyGILState_Release(gil_state);

		return pyobject_dict;
	}

	PyObject* m_convertVDictToPyDict(VDict src) {
		PyObject* pyobject_dict = PyDict_New();

		for (auto& it : src) {
			PyObject* key = PyUnicode_FromString(it.first.c_str());
			PyObject* value = m_convertVValueToPyObject(it.second);
			PyDict_SetItem(pyobject_dict, key, value);
		}

		return pyobject_dict;
	}

	PyObject* ConvertVListToPyList(VList src) {
		PyGILState_STATE gil_state = PyGILState_Ensure();

		PyObject* pyobject_list = m_convertVListToPyList(src);

		PyGILState_Release(gil_state);

		return pyobject_list;
	}

	PyObject* m_convertVListToPyList(VList src) {
		PyObject* pyobject_list = PyList_New(src.size());

		int nth = 0;

		for (auto& it : src) {
			PyObject* value;
			switch (it.type()) {
			case VValueType::int32:
				value = PyLong_FromLong((int)it);
				break;
			case VValueType::int64:
				value = PyLong_FromLongLong((int64)it);
				break;
			case VValueType::float32:
				value = PyFloat_FromDouble((double)it);
				break;
			case VValueType::kbool:
				value = PyBool_FromLong((int)it);
				break;
			case VValueType::string:
				value = PyUnicode_FromString(((string)it).c_str());
				break;
			case VValueType::list:
				value = m_convertVListToPyList((VList)it);
				break;
			case VValueType::dict:
				value = m_convertVDictToPyDict((VDict)it);
				break;
			case VValueType::shape:
				value = m_convertVShapeToPyList((VShape)it);
				break;
			default:
				throw std::runtime_error("Not implemented yet");
			}

			PyList_SET_ITEM(pyobject_list, nth++, value);
		}

		return pyobject_list;
	}

	PyObject* ConvertVStrListToPyList(VStrList src) {
		PyGILState_STATE gil_state = PyGILState_Ensure();

		PyObject* pyobject_list = PyList_New(src.size());

		int nth = 0;

		for (auto& it : src) {
			PyObject* value = PyUnicode_FromString(it.c_str());
			PyList_SET_ITEM(pyobject_list, nth++, value);
		}

		PyGILState_Release(gil_state);

		return pyobject_list;
	}

	PyObject* ConvertVShapeToPyList(VShape src) {
		PyGILState_STATE gil_state = PyGILState_Ensure();

		PyObject* pyobject_list = m_convertVShapeToPyList(src);

		PyGILState_Release(gil_state);

		return pyobject_list;
	}

	PyObject* m_convertVShapeToPyList(VShape src) {
		PyObject* pyobject_list = PyList_New(src.size());

		for (int64 n = 0; n < src.size(); n++) {
			PyObject* value = PyLong_FromLongLong((int64)src[n]);
			PyList_SET_ITEM(pyobject_list, n, value);
		}

		return pyobject_list;
	}

	PyObject* ConvertVectorToPyList(const std::vector<double>& src) {
		PyGILState_STATE gil_state = PyGILState_Ensure();

		PyObject* listObj = PyList_New(src.size());

		if (!listObj)
			throw std::logic_error("Unable to allocate memory for Python list");

		for (unsigned int i = 0; i < src.size(); ++i) {
			PyObject* num = PyFloat_FromDouble((double)src[i]);
			if (!num) {
				Py_DECREF(listObj);
				throw std::logic_error("Unable to allocate memory for Python list");
			}
			PyList_SET_ITEM(listObj, i, num);
		}

		PyGILState_Release(gil_state);
		return listObj;
	}

	PyObject* ConvertVectorToPyList(const std::vector<void*>& src) {
		PyGILState_STATE gil_state = PyGILState_Ensure();

		PyObject* listObj = PyList_New(src.size());

		if (!listObj)
			throw std::logic_error("Unable to allocate memory for Python list");

		for (unsigned int i = 0; i < src.size(); ++i) {
			PyObject* num = PyLong_FromVoidPtr((void*)src[i]);
			if (!num) {
				Py_DECREF(listObj);
				throw std::logic_error("Unable to allocate memory for Python list");
			}
			PyList_SET_ITEM(listObj, i, num);
		}

		PyGILState_Release(gil_state);
		return listObj;
	}

	PyObject* ConvertVectorToPyTuple(const std::vector<double>& src) {
		PyGILState_STATE gil_state = PyGILState_Ensure();

		PyObject* tuple = PyTuple_New(src.size());

		if (!tuple)
			throw std::logic_error("Unable to allocate memory for Python tuple");

		for (unsigned int i = 0; i < src.size(); ++i) {
			PyObject* num = PyFloat_FromDouble((double)src[i]);
			if (!num) {
				Py_DECREF(tuple);
				throw std::logic_error("Unable to allocate memory for Python tuple");
			}
			PyTuple_SET_ITEM(tuple, i, num);
		}

		PyGILState_Release(gil_state);
		return tuple;
	}

	PyObject* ConvertVectorToPyTuple(const std::vector<void*>& src) {
		PyGILState_STATE gil_state = PyGILState_Ensure();

		PyObject* tuple = PyTuple_New(src.size());

		if (!tuple)
			throw std::logic_error("Unable to allocate memory for Python tuple");

		for (unsigned int i = 0; i < src.size(); ++i) {
			PyObject* num = PyLong_FromVoidPtr((void*)src[i]);
			if (!num) {
				Py_DECREF(tuple);
				throw std::logic_error("Unable to allocate memory for Python tuple");
			}
			PyTuple_SET_ITEM(tuple, i, num);
		}

		PyGILState_Release(gil_state);
		return tuple;
	}

	PyObject* ConvertVector2DToPyTuple(const std::vector< std::vector<double> >& src) {
		PyGILState_STATE gil_state = PyGILState_Ensure();

		PyObject* tuple = PyTuple_New(src.size());

		if (!tuple)
			throw std::logic_error("Unable to allocate memory for Python tuple");

		for (unsigned int i = 0; i < src.size(); ++i) {
			PyObject* subTuple = NULL;
			try {
				subTuple = ConvertVectorToPyTuple(src[i]);
			}
			catch (std::logic_error& e) {
				throw e;
			}
			if (!subTuple) {
				Py_DECREF(tuple);
				throw std::logic_error("Unable to allocate memory for Python tuple of tuples");
			}
			PyTuple_SET_ITEM(tuple, i, subTuple);
		}

		PyGILState_Release(gil_state);
		return tuple;
	}

	PyObject* ConvertArray2Dpf32ToPyList(void* Array2Dpf32, int Array2Drow, int Array2Dcol)
	{
		PyGILState_STATE gil_state = PyGILState_Ensure();

		PyObject* PyObject_List		= PyList_New(3);
		PyObject* PyObject_pVoid	= PyLong_FromVoidPtr	(Array2Dpf32);
		PyObject* PyObject_Row		= PyLong_FromLong		(Array2Drow);
		PyObject* PyObject_Col		= PyLong_FromLong		(Array2Dcol);
		PyList_SET_ITEM(PyObject_List, 0, PyObject_pVoid);
		PyList_SET_ITEM(PyObject_List, 1, PyObject_Row);
		PyList_SET_ITEM(PyObject_List, 2, PyObject_Col);

		PyGILState_Release(gil_state);

		return PyObject_List;
	}

	PyObject* ConvertVShapeToPyTuple(const VShape* src)
	{
		PyGILState_STATE gil_state = PyGILState_Ensure();

		PyObject* listObj = PyTuple_New(src->size());

		if (!listObj)
			throw std::logic_error("Unable to allocate memory for Python list");

		for (unsigned int i = 0; i < src->size(); ++i) {
			PyObject* num = PyLong_FromVoidPtr((void*)(*src)[i]);
			if (!num) {
				Py_DECREF(listObj);
				throw std::logic_error("Unable to allocate memory for Python list");
			}
			PyTuple_SET_ITEM(listObj, i, num);
		}

		PyGILState_Release(gil_state);
		return listObj;
	}
};	// namespace kpy


////////////////////////////////////////////////////////////////////////////////////////////////////
// Python methods for handling python objects
////////////////////////////////////////////////////////////////////////////////////////////////////

void PrintPyObjectType(PyObject* py_object) {
	if (py_object == Py_None)            printf("None\n");
	else if (PyBool_Check(py_object))    printf("bool\n");
	else if (PyFloat_Check(py_object))   printf("float\n");
	else if (PyLong_Check(py_object))    printf("long\n");
	else if (PyList_Check(py_object))    printf("list\n");
	else if (PyTuple_Check(py_object))   printf("tuple\n");
	else if (PyBytes_Check(py_object))   printf("bytes\n");			// Python string.encode('utf-8') == C++ const char*
	else if (PyUnicode_Check(py_object)) printf("unicode\n");		// Python string
	else if (PyDict_Check(py_object))    printf("dictionary\n");
	else if (PyType_Check(py_object))    printf("type\n");
	else {
		// To avoid the bug of Python 3.9.x
		string type_name = py_object->ob_type->tp_name;
		std::transform(type_name.cbegin(), type_name.cend(), type_name.begin(), ::tolower);

		if (type_name == "float")
			printf("float\n");
		else
			printf("unknown type.\n");
	}
}

void PrintNumpy1DAsDouble(double* numpy_obj, int array_size) {
	double* ptr = numpy_obj;

	for (int i = 0; i < array_size; ++i) {
		printf("[%lld] %.15lf\n", (long long)i, *ptr++);
	}
}

void PrintListAndTupleAsDouble(PyObject* py_object) {
	PyObject* pIncoming = (PyObject*)py_object;

	CFUNCTYPE_PyListAndTuple_Size    cbPyListAndTuple_Size = NULL;
	CFUNCTYPE_PyListAndTuple_GetItem cbPyListAndTuple_GetItem = NULL;

	if (PyList_Check(pIncoming)) {
		cbPyListAndTuple_Size = PyList_Size;
		cbPyListAndTuple_GetItem = PyList_GetItem;
	}
	else if (PyTuple_Check(pIncoming)) {
		cbPyListAndTuple_Size = PyTuple_Size;
		cbPyListAndTuple_GetItem = PyTuple_GetItem;
	}
	else {
		fprintf(stderr, "error: Passed PyObject pointer was not a list or tuple.\n");
		return;
		//throw std::logic_error("Passed PyObject pointer was not a list or tuple.");
	}

	for (Py_ssize_t i = 0; i < cbPyListAndTuple_Size(pIncoming); ++i) {
		PyObject* value = cbPyListAndTuple_GetItem(pIncoming, i);
		double val = PyFloat_AsDouble(value);
		printf("[%lld] %.15lf\n", i, val);
	}
}

////////////////////////////////////////////////////////////////////////////////////////////////////
