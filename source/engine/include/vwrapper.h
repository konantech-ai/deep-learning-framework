#pragma once

#include "verrors.h"
#include "vtypes.h"
#include "vvalue.h"

#define VAL_THROW(x) { throw ValException(x, __FILE__, __LINE__); }

struct VExBuf {
	int m_size;
	char* m_contents;
};

class VWrapper {
public:
	VWrapper() {
		m_pTerm = NULL;
	}
	
	~VWrapper() {
		if (m_pTerm) {
			free(m_pTerm->m_contents);
			delete m_pTerm;
		}
	}

	const VExBuf* detach() {
		const VExBuf* pTerm = m_pTerm;
		m_pTerm = NULL;
		return pTerm;
	}

protected:
	VExBuf* m_pTerm;

	int m_serial_size(VValue vValue) {
		int size = 1;	// to store value type
		switch (vValue.type()) {
		case VValueType::none:
			break;
		case VValueType::kbool:
			size += 1;
			break;
		case VValueType::int32:
		case VValueType::float32:
			size += 4;
			break;
		case VValueType::int64:
			size += 8;
			break;
		case VValueType::string:
			size += sizeof(int) + (int)strlen(((string)vValue).c_str());
			break;
		case VValueType::list:
		{
			VList list = vValue;
			size += sizeof(int);
			for (auto& it : list) {
				size += m_serial_size(it);
			}
			break;
		}
		case VValueType::dict:
		{
			VDict dict = vValue;
			size += sizeof(int);
			for (auto& it:dict) {
				size += sizeof(int) + (int)strlen(it.first.c_str());
				size += m_serial_size(it.second);
			}
			break;
		}
		case VValueType::map:
		{
			VMap map = vValue;
			size += sizeof(int);
			for (auto& it : map) {
				size += sizeof(int);
				size += m_serial_size(it.second);
			}
			break;
		}
		case VValueType::shape:
		{
			VShape shape = vValue;
			size += ((int)shape.size() + 1) * sizeof(int);
			break;
		}
		case VValueType::object:
		{
			size += sizeof(VObjCore*);
			break;
		}
		case VValueType::pint32:
		case VValueType::pint64:
		case VValueType::pfloat:
		default:
			assert(0);
		}

		return size;
	}

	char* m_fill_int(char* cont, int nValue) {
		memcpy(cont, &nValue, sizeof(int));
		return cont + sizeof(int);
	}

	char* m_fill_string(char* cont, string sValue) {
		const char* str = sValue.c_str();
		int leng = (int)strlen(str);
		cont = m_fill_int(cont, leng);
		memcpy(cont, str, leng);
		return cont + leng;
	}

	char* m_fill_value(char* cont, VValue value) {
		VValueType type = value.type();
		*cont++ = (char)type;

		switch (type) {
		case VValueType::none:
			break;
		case VValueType::kbool:
			*cont++ = (bool)value ? 1 : 0;
			break;
		case VValueType::int32:
			cont = m_fill_int(cont, (int)value);
			break;
		case VValueType::float32:
		{
			float fValue = value;
			memcpy(cont, &fValue, sizeof(float));
			cont += sizeof(float);
			break;
		}
		case VValueType::int64:
		{
			int64 nValue = value;
			memcpy(cont, &nValue, sizeof(int64));
			cont += sizeof(int64);
			break;
		}
		case VValueType::string:
			cont = m_fill_string(cont, (string)value);
			break;
		case VValueType::list:
		{
			VList list = value;
			cont = m_fill_int(cont, (int)list.size());
			int nth = 0;
			for (auto& it: list) {
				cont = m_fill_value(cont, it); 
				nth++;
			}
			break;
		}
		case VValueType::dict:
		{
			VDict dict = value;
			cont = m_fill_int(cont, (int)dict.size());
			for (auto& it : dict) {
				cont = m_fill_string(cont, it.first);
				cont = m_fill_value(cont, it.second);
			}
			break;
		}
		case VValueType::map:
		{
			VMap map = value;
			cont = m_fill_int(cont, (int)map.size());
			for (auto& it : map) {
				cont = m_fill_int(cont, it.first);
				cont = m_fill_value(cont, it.second);
			}
			break;
		}
		case VValueType::shape:
		{
			VShape shape = value;
			cont = m_fill_int(cont, (int)shape.size());
			for (int n = 0; n < (int)shape.size(); n++) {
				cont = m_fill_int(cont, (int)shape[n]);
			}
			break;
		}
		case VValueType::object:
		{
			VObjCore* pCore = value;
			pCore->clone_core();
			memcpy(cont, &pCore, sizeof(void*));
			cont += sizeof(void*);
			break;
		}
		case VValueType::pint32:
		case VValueType::pint64:
		case VValueType::pfloat:
		default:
			assert(0);
		}

		return cont;
	}

	static int ms_get_int(char* contents, int& pos) {
		int nValue;
		memcpy(&nValue, contents+pos, sizeof(int));
		pos += sizeof(int);
		return nValue;
	}

	static string ms_get_string(char* contents, int& pos) {
		int leng = ms_get_int(contents, pos);
		string sValue = string(contents + pos, leng);
		pos += leng;
		return sValue;
	}

	static VValue ms_get_value(char* contents, int& pos) {
		VValueType type = (VValueType)(int)contents[pos++];
		VValue value;

		switch (type) {
		case VValueType::none:
			break;
		case VValueType::kbool:
			value = VValue((bool)contents[pos++]);
			break;
		case VValueType::int32:
		{
			int nValue;
			memcpy(&nValue, contents + pos, sizeof(int));
			pos += sizeof(int);
			value = nValue;
			break;
		}
		case VValueType::float32:
		{
			float fValue;
			memcpy(&fValue, contents + pos, sizeof(float));
			pos += sizeof(float);
			value = fValue;
			break;
		}
		case VValueType::int64:
		{
			int64 nValue;
			memcpy(&nValue, contents + pos, sizeof(int64));
			pos += sizeof(int64);
			value = nValue;
			break;
		}
		case VValueType::string:
			value = ms_get_string(contents, pos);
			break;
		case VValueType::list:
		{
			VList list;
			int size = ms_get_int(contents, pos);
			for (int n = 0; n < size; n++) {
				list.push_back(ms_get_value(contents, pos));
			}
			value = list;
			break;
		}
		case VValueType::dict:
		{
			VDict dict;
			int size = ms_get_int(contents, pos);
			for (int n = 0; n < size; n++) {
				string key = ms_get_string(contents, pos);
				dict[key] = ms_get_value(contents, pos);
			}
			value = dict;
			break;
		}
		case VValueType::map:
		{
			VMap map;
			int size = ms_get_int(contents, pos);
			for (int n = 0; n < size; n++) {
				int key = ms_get_int(contents, pos);
				map[key] = ms_get_value(contents, pos);
			}
			value = map;
			break;
		}
		case VValueType::shape:
		{
			VShape shape;
			int size = ms_get_int(contents, pos);
			for (int n = 0; n < size; n++) {
				int64 axis_size = (int64) ms_get_int(contents, pos);
				shape = shape.append(axis_size);
			}
			value = shape;
			//VShape shape;
			/*int size = ms_get_int(contents, pos);
			int* axis_size = new int[size];
			if (axis_size == NULL) VAL_THROW(VERR_HOSTMEM_ALLOC_FAILURE);
			for (int n = 0; n < size; n++) {
				axis_size[n] = ms_get_int(contents, pos);
				//shape = shape.append(axis_size);
			}
			VShape shape(size, axis_size);
			value = shape;
			delete[] axis_size;
			*/
			break;
		}
		case VValueType::object:
		{
			VObjCore* pCore;
			memcpy(&pCore, contents + pos, sizeof(void*));
			pos += sizeof(void*);
			value = pCore; // ->destroy();
			break;
		}
		case VValueType::pint32:
		case VValueType::pint64:
		case VValueType::pfloat:
		default:
			assert(0);
		}

		return value;
	}
};

class VDictWrapper : public VWrapper {
public:
	VDictWrapper(VDict dict) {
		int size = sizeof(int);

		for (auto& it : dict) {
			size += sizeof(int) + (int)strlen(it.first.c_str());
			size += m_serial_size(it.second);
		}

		m_pTerm = new VExBuf();
		m_pTerm->m_size = size;
		m_pTerm->m_contents = (char*)malloc(size);

		if (m_pTerm->m_contents == NULL) VAL_THROW(VERR_HOSTMEM_ALLOC_FAILURE);

		char* cont = m_pTerm->m_contents;

		cont = m_fill_int(cont, (unsigned short int)dict.size());

		for (auto& it : dict) {
			cont = m_fill_string(cont, it.first);
			cont = m_fill_value(cont, it.second);
		}
	}

	static VDict unwrap(const VExBuf* pBuf) {
		VDict dict;
		int pos = 0;
		int dicSize = ms_get_int(pBuf->m_contents, pos);

		for (int64 n = 0; n < dicSize; n++) {
			string key = ms_get_string(pBuf->m_contents, pos);
			VValue value = ms_get_value(pBuf->m_contents, pos);
			dict[key] = value;
		}

		return dict;
	}
};

class VListWrapper : public VWrapper {
public:
	VListWrapper(VList list) {
		int size = sizeof(int);

		for (auto& it : list) {
			size += m_serial_size(it);
		}

		m_pTerm = new VExBuf();
		m_pTerm->m_size = size;
		m_pTerm->m_contents = (char*)malloc(size);

		if (m_pTerm->m_contents == NULL) VAL_THROW(VERR_HOSTMEM_ALLOC_FAILURE);

		char* cont = m_pTerm->m_contents;

		cont = m_fill_int(cont, (unsigned short int)list.size());

		for (auto& it : list) {
			cont = m_fill_value(cont, it);
		}
	}

	static VList unwrap(const VExBuf* pBuf) {
		VList list;
		int pos = 0;
		int listSize = ms_get_int(pBuf->m_contents, pos);

		for (int64 n = 0; n < listSize; n++) {
			VValue value = ms_get_value(pBuf->m_contents, pos);
			list.push_back(value);
		}

		return list;
	}
};

class VShapeWrapper : public VWrapper {
public:
	VShapeWrapper(VShape shape) {
		int size = (1 + (int)shape.size()) * sizeof(int);

		m_pTerm = new VExBuf();
		m_pTerm->m_size = size;
		m_pTerm->m_contents = (char*)malloc(size);

		if (m_pTerm->m_contents == NULL) VAL_THROW(VERR_HOSTMEM_ALLOC_FAILURE);

		char* cont = m_pTerm->m_contents;

		cont = m_fill_int(cont, (int)shape.size());

		for (int64 n = 0; n < shape.size(); n++) {
			cont = m_fill_int(cont, (int)shape[n]);
		}
	}

	static VShape unwrap(const VExBuf* pBuf) {
		VShape shape;
		int pos = 0;
		int shapeSize = ms_get_int(pBuf->m_contents, pos);

		for (int64 n = 0; n < shapeSize; n++) {
			int64 axis_size = (int64)ms_get_int(pBuf->m_contents, pos);
			shape = shape.append(axis_size);
		}

		return shape;
	}
};

typedef int VCbCustomModuleExec(void* pInst, void* pAux, time_t time, VHandle hModule, const VExBuf* pXsBuf, const VExBuf** ppYsBuf);
typedef int VCbFreeReportBuffer(void* pInst, void* pAux, const VExBuf* pResultBuf);
