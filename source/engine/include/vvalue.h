#pragma once

#include <assert.h>
#include <stdio.h>
#include <stdarg.h>
#include <algorithm>

#include "vtypes.h"
#include "verrors.h"

class ValException {
public:
	ValException(int nErrCode, string file, int line) {
		m_nErrCode = nErrCode;
		m_file = file;
		m_line = line;
		std::replace(m_file.begin(), m_file.end(), '\\', '/');
	}
	~ValException() { }
	int m_nErrCode;
	string m_file;
	int m_line;
};

inline int64 kaxis_mod(int64 a, int64 b) { return b ? (a % b + b) % b : 0; }

class VObjCore {
#if defined(V_DEBUG_OBJ_LEAK)
protected:
	VObjCore(VObjType type) {
		m_type = type;
		
		ms_ref_mutex.lock();

		m_nth = ms_nNextId++;
		m_nRefCnt = 1;
		m_domain = ms_nDomain;
		ms_nInstCount++;
		//ms_instIds.insert(m_nth);
		ms_instAlive[m_nth] = this;

		ms_ref_mutex.unlock();

		if (m_domain == ms_nDebugDomain && m_nth == ms_nDebugNth) {
			printf("[%d] %d Obj#%d-%d created %s\n", ms_nth_nth++, m_nRefCnt, ms_nDebugDomain, m_nth, ms_nDomain ? " on client" : "");
		}
	}

public:
	void destroy() {
		if (this == NULL) return;
		if (m_nth == 0 && (int)m_type == 0) {
			return;	// model.__str__() 확장으로 모듈 정보를 서버에서 얻어오자 등장한 유령같은 객체, 정체 파악 필요
			// delete 되면서 프로그램이 죽기 때문에 일단 살려놓고 삺펴가기로 함
		}
		ms_ref_mutex.lock();
		if (m_domain == ms_nDebugDomain && m_nth == ms_nDebugNth) {
			printf("[%d] %d Obj#%d-%d destroyed%s\n", ms_nth_nth++, m_nRefCnt - 1, ms_nDebugDomain, m_nth, ms_nDomain ? " on client" : "");
		}

		//if (m_domain == ms_nDomain && ms_instIds.find(m_nth) == ms_instIds.end()) {
		if (m_domain == ms_nDomain && ms_instAlive.find(m_nth) == ms_instAlive.end()) {
			printf("Object overkilled: %d\n", ms_instDead[this]);
			ms_ref_mutex.unlock();
			return;
		}

		if (--m_nRefCnt <= 0) {
			if (m_domain != ms_nDomain) {
				static bool dump = false;
				// 핸들 전달을 위한 VValue(int64) 원소: bad domaon 삭제이지만 문제 발생 않음
				if (dump) {
					int nDead = ms_instDead[this];
					printf("Object deletion in bad domain(%d): OBJ-%d(domain %d)\n", ms_nDomain, nDead, m_domain);
					printf("(더 이상 유사 현상 관련 메시지 출력 안하지만 반드시 디버깅 요망)\n");
					printf("(VDict 등의 구조체가 관문 통과시 래핑 변환 처리 거치지 않아 다른 모듈에서 삭제되는 탓에 발생)\n\n");
					dump = false;
				}
				//throw ValException(VERR_UNDEFINED, __FILE__, __LINE__);
			}

			// domain 달라 삭제가 위험을 초래하는 경우에 대해서는 철저한 디버깅 필요
			// 단, 전체적인 처리과정 확인을 위해 임시로 삭제하지 않고 방치해 예외발생을막아보기로 한다.
			if (m_domain == ms_nDomain) {
				ms_nInstCount--;
				//ms_instIds.erase(ms_instIds.find(m_nth));
				//ms_instMap[this] = m_nth;
				ms_instAlive.erase(ms_instAlive.find(m_nth));
				ms_instDead[this] = m_nth;
				ms_ref_mutex.unlock();
				if (m_domain == ms_nDebugDomain && m_nth == ms_nDebugNth) {
					printf("[%d] %d Obj#%d-%d deleted%s\n", ms_nth_nth++, m_nRefCnt, ms_nDebugDomain, m_nth, ms_nDomain ? " on client" : "");
				}
				delete this;
				return;
			}
		}

		ms_ref_mutex.unlock();
	}

	void destroy_handle() {
		if (this == NULL) return;
		ms_ref_mutex.lock();
		if (m_domain == ms_nDebugDomain && m_nth == ms_nDebugNth) {
			printf("[%d] %d Obj#%d-%d destroyed%s by handle\n", ms_nth_nth++, m_nRefCnt - 2, ms_nDebugDomain, m_nth, ms_nDomain ? " on client" : "");
		}

		//if (m_domain == ms_nDomain && ms_instIds.find(m_nth) == ms_instIds.end()) {
		if (m_domain == ms_nDomain && ms_instAlive.find(m_nth) == ms_instAlive.end()) {
			printf("Object overkilled: %d\n", ms_instDead[this]);
			ms_ref_mutex.unlock();
			return;
		}

		m_nRefCnt -= 2;

		if (m_nRefCnt <= 0) {
			if (m_domain != ms_nDomain) {
				static bool dump = true;
				// 핸들 전달을 위한 VValue(int64) 원소: bad domaon 삭제이지만 문제 발생 않음
				if (dump) {
					int nDead = ms_instDead[this];
					printf("Object Handle deletion in bad domain(%d): OBJ-%d(domain %d)\n", ms_nDomain, nDead, m_domain);
					printf("(더 이상 유사 현상 관련 메시지 출력 안하지만 반드시 디버깅 요망\n");
					dump = false;
				}
				//throw ValException(VERR_UNDEFINED, __FILE__, __LINE__);
			}

			// domain 달라 삭제가 위험을 초래하는 경우에 대해서는 철저한 디버깅 필요
			// 단, 전체적인 처리과정 확인을 위해 임시로 삭제하지 않고 방치해 예외발생을막아보기로 한다.
			if (m_domain == ms_nDomain) {
				ms_nInstCount--;
				//ms_instIds.erase(ms_instIds.find(m_nth));
				//ms_instMap[this] = m_nth;
				ms_instAlive.erase(ms_instAlive.find(m_nth));
				ms_instDead[this] = m_nth;
				ms_ref_mutex.unlock();
				if (m_domain == ms_nDebugDomain && m_nth == ms_nDebugNth) {
					printf("[%d] %d Obj#%d-%d deleted by handle%s\n", ms_nth_nth++, m_nRefCnt, ms_nDebugDomain, m_nth, ms_nDomain ? " on client" : "");
				}
				delete this;
				return;
			}
		}

		ms_ref_mutex.unlock();
	}

	int getDebugId() { return m_nth; }

public:
	VObjCore* clone_core() {
		if (this == NULL) return NULL;

		ms_ref_mutex.lock();
		//if (m_domain == ms_nDomain && ms_instIds.find(m_nth) == ms_instIds.end()) {
		if (m_domain == ms_nDomain && ms_instAlive.find(m_nth) == ms_instAlive.end()) {
			printf("Object jombie touched: %d\n", ms_instDead[this]);
			ms_ref_mutex.unlock();
			return this;
		}

		m_nRefCnt++;
		if (m_domain == ms_nDebugDomain && m_nth == ms_nDebugNth) {
			printf("[%d] %d Obj#%d-%d detached%s\n", ms_nth_nth++, m_nRefCnt, ms_nDebugDomain, m_nth, ms_nDomain?" on client":"");
		}
		ms_ref_mutex.unlock();
		return this;
	}
	VObjCore* clone_handle() {
		if (this == NULL) return NULL;

		ms_ref_mutex.lock();
		//if (m_domain == ms_nDomain && ms_instIds.find(m_nth) == ms_instIds.end()) {
		if (m_domain == ms_nDomain && ms_instAlive.find(m_nth) == ms_instAlive.end()) {
			printf("Object jombie touched: %d\n", ms_instDead[this]);
			ms_ref_mutex.unlock();
			return this;
		}

		m_nRefCnt += 2;
		if (m_domain == ms_nDebugDomain && m_nth == ms_nDebugNth) {
			printf("[%d] %d Obj#%d-%d detached to handle%s\n", ms_nth_nth++, m_nRefCnt, ms_nDebugDomain, m_nth, ms_nDomain ? " on client" : "");
		}
		ms_ref_mutex.unlock();
		return this;
	}
#else
protected:
	VObjCore(VObjType type) {
		m_type = type;
		m_nRefCnt = 1;
		m_nth = ms_nNextId++;
	}
	
public:
	void destroy() {
		if (this == NULL) return;
		ms_ref_mutex.lock();
		--m_nRefCnt;
		ms_ref_mutex.unlock();
		if (m_nRefCnt <= 0) delete this;
	}

	void destroy_handle() {
		if (this == NULL) return;
		ms_ref_mutex.lock();
		m_nRefCnt -= 2;
		ms_ref_mutex.unlock();
		if (m_nRefCnt <= 0) delete this;
	}

	//int getDebugId() { return -1; }

public:
	VObjCore* clone_core() {
		if (this == NULL) return NULL;
		ms_ref_mutex.lock();
		m_nRefCnt++;
		ms_ref_mutex.unlock();
		return this;
	}
	VObjCore* clone_handle() {
		if (this == NULL) return NULL;
		ms_ref_mutex.lock();
		m_nRefCnt += 2;
		ms_ref_mutex.unlock();
		return this;
	}
#endif

public:
	int getNth() { return this ? m_nth : -1; }
	int getRefCnt() { return this ? m_nRefCnt : -1; }
	void incRefCnt() { m_nRefCnt++; }
	VObjType getType() { return m_type; }
	virtual string desc() { return "no info"; }

protected:
	virtual ~VObjCore() {}

private:
	friend class VValue;

private:
	int m_nth;
	int m_nRefCnt;
	VObjType m_type;
	static mutex ms_ref_mutex;
	static int ms_nNextId;

#if defined(V_DEBUG_OBJ_LEAK)
	friend void dumpObjectUsage(string title);
	static void DumpUsage();

	int m_domain;

	static int ms_nDebugNth;
	static int ms_nDebugDomain;
	static int ms_nInstCount;
	static int ms_nDomain;
	static int ms_nth_nth;
	//static set<int> ms_instIds;
	static map<int, VObjCore*> ms_instAlive;
	static map<VObjCore*, int> ms_instDead;
#endif
};

class VList;
class VDict;
class VMap;
class VShape;
//class VTensor;

class VListCore;
class VDictCore;
class VMapCore;
class VShapeCore;
class VTensorCore;

class VValueCore : public VObjCore {
protected:
	VValueType m_type;
	union value_union {
		int m_int32;
		int* m_pint32;
		int64 m_int64;
		int64* m_pint64;
		float m_float;
		float* m_pfloat;
		bool m_bool;
		VObjCore* m_pCore;
		VShapeCore* m_pShapeCore;
	} m_value;
	string m_string;

	friend class VValue;

protected:
	VValueCore() : VObjCore(VObjType::value) {
		m_type = VValueType::none;
		m_value.m_int64 = 0;
	}
	~VValueCore();

	VValueCore* clone() { return (VValueCore*)clone_core(); }
};

class VValue {
protected:
	VValueCore* m_core;

public:
	VValue() { m_core = new VValueCore(); }
	VValue(const VValue& src) { m_core = src.m_core->clone();  }
	VValue(int nValue) { m_core = new VValueCore(); m_core->m_type = VValueType::int32; m_core->m_value.m_int32 = nValue; }
	VValue(int64 nValue) {
		m_core = new VValueCore();
		m_core->m_type = VValueType::int64;
		m_core->m_value.m_int64 = nValue;
	}
	VValue(int* pnValue) {
		m_core = new VValueCore();
		m_core->m_type = VValueType::pint32;
		m_core->m_value.m_pint32 = pnValue;
	}
	VValue(int64* pnValue) { m_core = new VValueCore(); m_core->m_type = VValueType::pint64; m_core->m_value.m_pint64 = pnValue; }
	VValue(string sValue) { m_core = new VValueCore(); m_core->m_type = VValueType::string; m_core->m_string = sValue; }
	VValue(const char* sValue) { m_core = new VValueCore(); m_core->m_type = VValueType::string; m_core->m_string = sValue; }
	VValue(float fValue) { m_core = new VValueCore(); m_core->m_type = VValueType::float32; m_core->m_value.m_float = fValue; }
	VValue(float* pfValue) { m_core = new VValueCore(); m_core->m_type = VValueType::pfloat; m_core->m_value.m_pfloat = pfValue; }
	VValue(double fValue) { m_core = new VValueCore(); m_core->m_type = VValueType::float32; m_core->m_value.m_float = (float)fValue; }
	VValue(bool bValue) { m_core = new VValueCore(); m_core->m_type = VValueType::kbool; m_core->m_value.m_bool = bValue; }
	VValue(VList lValue);
	VValue(VStrList lValue); // VList로 변환하여 저장함
	VValue(VDict dValue);
	VValue(VMap dValue);
	VValue(VShape sValue);
	//VValue(VTensor sValue);
	VValue(VObjCore* pValue);

	~VValue() { m_core->destroy(); }

	VValue& operator =(const VValue& src) {
		if (&src != this && m_core != src.m_core) {
			m_core->destroy();
			m_core = src.m_core->clone();
		}
		return *this;
	}

	//VObjCore* cloneCoreValue() { return m_core->m_value.m_pCore->clone_core(); }

	VValue& operator =(int nValue) { m_core->destroy(); m_core = new VValueCore(); m_core->m_type = VValueType::int32; m_core->m_value.m_int32 = nValue; return *this; }
	VValue& operator =(int64 nValue) { m_core->destroy(); m_core = new VValueCore(); m_core->m_type = VValueType::int64; m_core->m_value.m_int64 = nValue; return *this; }
	VValue& operator =(int* pnValue) { m_core->destroy(); m_core = new VValueCore(); m_core->m_type = VValueType::pint32; m_core->m_value.m_pint32 = pnValue; return *this; }
	VValue& operator =(int64* pnValue) { m_core->destroy(); m_core = new VValueCore(); m_core->m_type = VValueType::pint64; m_core->m_value.m_pint64 = pnValue; return *this; }
	VValue& operator =(string sValue) { m_core->destroy(); m_core = new VValueCore(); m_core->m_type = VValueType::string; m_core->m_string = sValue; return *this; }
	VValue& operator =(const char* sValue) { m_core->destroy(); m_core = new VValueCore(); m_core->m_type = VValueType::string; m_core->m_string = sValue; return *this; }
	VValue& operator =(float fValue) { m_core->destroy(); m_core = new VValueCore(); m_core->m_type = VValueType::float32; m_core->m_value.m_float = fValue; return *this; }
	VValue& operator =(double fValue) { m_core->destroy(); m_core = new VValueCore(); m_core->m_type = VValueType::float32; m_core->m_value.m_float = (float)fValue; return *this; }
	VValue& operator =(float* pfValue) { m_core->destroy(); m_core = new VValueCore(); m_core->m_type = VValueType::pfloat; m_core->m_value.m_pfloat = pfValue; return *this; }
	VValue& operator =(bool bValue) { m_core->destroy(); m_core = new VValueCore(); m_core->m_type = VValueType::kbool; m_core->m_value.m_bool = bValue; return *this; }
	VValue& operator =(VList lValue);
	//VValue& operator =(VList& lValue);
	VValue& operator =(VDict dValue);
	VValue& operator =(VMap dValue);
	//VValue& operator =(VTensor tValue);
	VValue& operator =(VShape sValue);
	VValue& operator =(VObjCore* hValue);

	bool operator ==(const VValue& vValue) const;
	bool operator ==(const bool sValue) const;
	bool operator ==(int nValue) const;
	bool operator ==(int64 nValue) const;
	bool operator ==(int64* pnValue) const;
	bool operator ==(string sValue) const;
	bool operator ==(const char* sValue) const;
	bool operator ==(float fValue) const;
	bool operator ==(float* pfValue) const;
	bool operator ==(VList lValue) const;
	bool operator ==(VDict dValue) const;
	bool operator ==(VMap dValue) const;
	//bool operator ==(VTensor tValue) const;
	bool operator ==(VShape sValue) const;
	bool operator ==(VObjCore* hValue) const;

	bool operator !=(const VValue& vValue) const;
	bool operator !=(const bool sValue) const;
	bool operator !=(int nValue) const;
	bool operator !=(int64 nValue) const;
	bool operator !=(int64* pnValue) const;
	bool operator !=(string sValue) const;
	bool operator !=(const char* sValue) const;
	bool operator !=(float fValue) const;
	bool operator !=(float* pfValue) const;
	bool operator !=(VList lValue) const;
	bool operator !=(VDict dValue) const;
	bool operator !=(VMap dValue) const;
	bool operator !=(VShape sValue) const;
	bool operator !=(VObjCore* hValue) const;

	VValueType type() const { return m_core->m_type; }

	bool is_none() const { return type() == VValueType::none; }
	bool is_string() const { return type() == VValueType::string; }
	bool is_float() const { return type() == VValueType::float32; }
	bool is_int() const { return type() == VValueType::int32 || type() == VValueType::int64; }
	bool is_int32() const { return type() == VValueType::int32; }
	bool is_int64() const { return type() == VValueType::int64; }
	bool is_bool() const { return type() == VValueType::kbool; }
	bool is_float_ptr() const { return type() == VValueType::pfloat; }
	bool is_int32_ptr() const { return type() == VValueType::pint32; }
	bool is_int64_ptr() const { return type() == VValueType::pint64; }
	bool is_list() const { return type() == VValueType::list; }
	bool is_dict() const { return type() == VValueType::dict; }
	bool is_shape() const { return type() == VValueType::shape; }
	bool is_object() const { return type() == VValueType::object; }
	bool is_tensor() const {
		if (type() != VValueType::object) return false;
		VObjCore* pObj = (VObjCore*)(*this);
		if (pObj->getType() == VObjType::Tensor) return true;
		return false;
	}

	string to_string() const;

	operator int() const;
	operator int64() const;
	operator int* () const;
	operator int64* () const;
	operator float() const;
	operator float* () const;
	operator bool() const;
	operator string() const;
	operator VList();
	operator VDict();
	operator VMap();
	//operator VTensor();
	operator VShape();
	operator VObjCore* () const;

	string desc() const;

};

typedef std::map<std::string, VValue>::iterator VDictIter;
typedef std::map<int, VValue>::iterator VMapIter;
typedef std::vector<VValue>::iterator VListIter;
typedef std::initializer_list<VValue>::iterator _initIt;
typedef std::initializer_list<std::initializer_list<VValue>>::iterator _initIt2;
typedef std::initializer_list<int64>::iterator _initIt_n;
typedef std::initializer_list<std::initializer_list<int64>>::iterator _initIt2_n;

class VListCore : public VObjCore {
	std::vector<VValue> m_list;

	friend class VList;
	friend class VValue;
	friend class VValueCore;

	VListCore() : VObjCore(VObjType::list) {}
	~VListCore() {}
	VListCore* clone() { return (VListCore*)clone_core(); }
};

class VList {
protected:
	VListCore* m_core;
	friend class VValue;
	friend class VOptimizerCore; // for debugging

public:
	VList() { m_core = new VListCore(); }
	VList(const VList& src) { m_core = src.m_core->clone(); }
	VList(VListCore* core) { m_core = core->clone(); }
	VList(std::initializer_list<VValue> list) {
		m_core = new VListCore();
		for (VValue ax : list) {
			m_core->m_list.push_back(ax);
		}
	}
	VList(std::initializer_list<int> list) {
		m_core = new VListCore();
		for (int ax : list) {
			m_core->m_list.push_back((int64)ax);
		}
	}
	VList(std::initializer_list<int64> list) {
		m_core = new VListCore();
		for (int64 ax : list) {
			m_core->m_list.push_back(ax);
		}
	}
	VList(std::initializer_list<float> list) {
		m_core = new VListCore();
		for (float ax : list) {
			m_core->m_list.push_back(ax);
		}
	}
	VList(std::initializer_list<std::initializer_list<int64>> list) {
		m_core = new VListCore();
		for (_initIt2_n it2 = list.begin(); it2 != list.end(); it2++) {
			std::initializer_list<int64> term = *it2;
			_initIt_n it = term.begin();
			VList v(term);
			m_core->m_list.push_back(v);
		}
	}
	~VList() { m_core->destroy(); }

	VList& operator =(const VList& src) {
		if (&src != this && m_core != src.m_core) {
			m_core->destroy();
			m_core = src.m_core->clone();
		}
		return *this;
	}

	//VListCore* detach() { return m_core->detach(); }

	int64 size() const { return (int64)m_core->m_list.size(); }

	void clear() { m_core->m_list.clear(); }
	void push_back(VValue value) { m_core->m_list.push_back(value); }
	void erase(VListIter it) { m_core->m_list.erase(it); }

	VValue& operator [](int64 nIndex) { return m_core->m_list[nIndex]; }
	VValue operator [](int64 nIndex) const { return m_core->m_list[nIndex]; }

	VListIter begin() const { return m_core->m_list.begin(); }
	VListIter end() const { return m_core->m_list.end(); }

	VListIter find(VValue value) const { return std::find(m_core->m_list.begin(), m_core->m_list.end(), value); }
	bool find_string(string sValue) const;

	string desc() const {
		string desc, delimeter = "[";
		for (int64 n = 0; n < size(); n++) { desc += delimeter; delimeter = ",";  desc += m_core->m_list[n].desc(); }
		return desc + "]";
	}

	string desc_lines() const {
		string desc = "[", delimeter = "";
		for (int64 n = 0; n < size(); n++) { desc += delimeter; delimeter = ",\n ";  desc += m_core->m_list[n].desc(); }
		return desc + "]";
	}
};

class VDictCore : public VObjCore {
	std::map<string, VValue> m_dict;

	friend class VDict;
	friend class VValue;
	friend class VValueCore;

	VDictCore() : VObjCore(VObjType::dict) {}
	~VDictCore() {}
	VDictCore* clone() { return (VDictCore*)clone_core(); }
};

class VDict {
public:
	VDict() { m_core = new VDictCore(); }
	VDict(const VDict& src) { m_core = src.m_core->clone(); }
	VDict(VDictCore* core) { m_core = core->clone(); }
	VDict(std::initializer_list<VValue> list) {
		m_core = new VDictCore();
		for (_initIt it = list.begin(); it != list.end(); ) {
			string k = (string)(VValue)*it; it++;
			VValue v = *it; it++;
			m_core->m_dict[k] = v;
		}
	}
	VDict(std::initializer_list<std::initializer_list<VValue>> list) {
		m_core = new VDictCore();
		for (_initIt2 it2 = list.begin(); it2 != list.end(); it2++) {
			std::initializer_list<VValue> term = *it2;
			_initIt it = term.begin();
			string k = (string)(VValue)*it; it++;
			VValue v = *it;
			m_core->m_dict[k] = v;
		}
	}
	~VDict() { m_core->destroy(); }


	VDict& operator =(const VDict& src) {
		if (m_core != src.m_core) {
			m_core->destroy();
			m_core = src.m_core->clone();
		}

		return *this;
	}

	bool operator ==(VDict dValue) const;
	bool operator !=(VDict dValue) const;

	VDictCore* getCore() { return (VDictCore*)m_core; }
	VDictCore* cloneCore() { return (VDictCore*)m_core->clone_core(); }

	void freeClone() { m_core->destroy(); }

	//VDictCore* getCore() { return m_core; }
	//void destroy() { m_core->destroy(); }

	int64 size() const { return (int64)m_core->m_dict.size(); }

	void clear() { m_core->m_dict.clear(); }
	void erase(string sKey) { m_core->m_dict.erase(sKey); }

	VValue& operator [](string sKey) { return m_core->m_dict[sKey]; }
	VValue operator [](string sKey) const { return m_core->m_dict[sKey]; }

	VDictIter begin() const { return m_core->m_dict.begin(); }
	VDictIter end() const { return m_core->m_dict.end(); }
	VDictIter nth(int n) const {
		VDictIter iter = m_core->m_dict.begin();
		while (--n >= 0) iter++;
		return iter;
	}

	VDictIter find(string sKey) const { return m_core->m_dict.find(sKey); }

	string desc() const {
		if (size() == 0) return "{}";
		string desc, delimeter = "{";
		for (VDictIter it = begin(); it != end(); it++) {
			desc += delimeter;
			delimeter = ",";
			desc += "'" + it->first + "':" + it->second.desc(); }
		return desc + "}";
	}

protected:
	VDictCore* m_core;
	friend class VValue;
	friend class VOptimizerCore; // for debugging
};

class VMapCore : public VObjCore {
	std::map<int, VValue> m_map;

	friend class VMap;
	friend class VValue;
	friend class VValueCore;

	VMapCore() : VObjCore(VObjType::map) {}
	~VMapCore() {}
	VMapCore* clone() { return (VMapCore*)clone_core(); }
};

class VMap {
public:
	VMap() { m_core = new VMapCore(); }
	VMap(const VMap& src) { m_core = src.m_core->clone(); }
	VMap(VMapCore* core) { m_core = (VMapCore *)core->clone(); }
	VMap(std::initializer_list<VValue> list) {
		m_core = new VMapCore();
		for (_initIt it = list.begin(); it != list.end(); ) {
			int k = (int)(VValue)*it; it++;
			VValue v = *it; it++;
			m_core->m_map[k] = v;
		}
	}
	VMap(std::initializer_list<std::initializer_list<VValue>> list) {
		m_core = new VMapCore();
		for (_initIt2 it2 = list.begin(); it2 != list.end(); it2++) {
			std::initializer_list<VValue> term = *it2;
			_initIt it = term.begin();
			int k = (int)(VValue)*it; it++;
			VValue v = *it;
			m_core->m_map[k] = v;
		}
	}
	~VMap() { m_core->destroy(); }

	VMap& operator =(const VMap& src) {
		if (&src != this && m_core != src.m_core) {
			m_core->destroy();
			m_core = src.m_core->clone();
		}
		return *this;
	}

	//VMapCore* detach() { m_core->m_nRefCnt++; return m_core; }

	int64 size() const { return (int64)m_core->m_map.size(); }

	void clear() { m_core->m_map.clear(); }
	void erase(int nKey) { m_core->m_map.erase(nKey); }

	VValue& operator [](int nKey) { return m_core->m_map[nKey]; }
	VValue operator [](int nKey) const { return m_core->m_map[nKey]; }

	VMapIter begin() const { return m_core->m_map.begin(); }
	VMapIter end() const { return m_core->m_map.end(); }
	VMapIter nth(int n) const {
		VMapIter iter = m_core->m_map.begin();
		while (--n >= 0) iter++;
		return iter;
	}

	VMapIter find(int nKey) const { return m_core->m_map.find(nKey); }

	string desc() const {
		string desc, delimeter = "{";
		for (VMapIter it = begin(); it != end(); it++) { desc += delimeter; delimeter = ",";  desc += to_string(it->first) + ":" + it->second.desc(); }
		return desc + "}";
	}

protected:
	VMapCore* m_core;
	friend class VValue;
};


class VShapeCore {
	std::vector<int64> m_shape;
	int m_nRefCnt;

	friend class VShape;
	friend class VValue;
	friend class VValueCore;

	VShapeCore() {
		m_nRefCnt = 1;
	}
	~VShapeCore() {
	}
	void destroy() {
		if (--m_nRefCnt <= 0) delete this;
	}
	VShapeCore* clone() {
		m_nRefCnt++;
		return this;
	}
};

class VShape {
public:
	VShape() { m_core = new VShapeCore(); }
	VShape(const VShape& src) {
		m_core = src.m_core->clone();
	}
	VShape(VShapeCore* core) {
		m_core = core->clone();
	}
	VShape(VList list) { m_core = new VShapeCore(); for (int64 n = 0; n < list.size(); n++) m_core->m_shape.push_back((int64)list[n]); }
	VShape(int64 dim, int64* ax_size) { m_core = new VShapeCore(); for (int64 n = 0; n < dim; n++) m_core->m_shape.push_back(ax_size[n]); }
	VShape(int dim, int* ax_size) { m_core = new VShapeCore(); for (int64 n = 0; n < dim; n++) m_core->m_shape.push_back(ax_size[n]); }
	VShape(std::initializer_list<int64> list) { m_core = new VShapeCore(); for (int64 ax : list) m_core->m_shape.push_back(ax); }
	~VShape() { m_core->destroy(); }

	VShape& operator =(const VShape& src) {
		if (&src != this && m_core != src.m_core) {
			m_core->destroy();
			m_core = src.m_core->clone();
		}
		return *this;
	}

	int64 size() const { return (int64)m_core->m_shape.size(); }
	int64 total_size() const { if (size() == 0) return 1; int64 prod = 1; for (int64 ax : m_core->m_shape) prod *= ax; return prod; }
	int64 valid_size() const { if (size() == 0) return 0; int64 prod = 1; for (int64 ax : m_core->m_shape) prod *= ax; return prod; }
	int64 head_size(int64 len = 1) const { int64 prod = 1; for (int64 n = 0; n < len; n++) prod *= m_core->m_shape[n]; return prod; }
	int64 seed_size() const { int64 prod = 1; for (int64 ax : m_core->m_shape) prod *= ax; return prod; }

	void copyInto(VShape shape) {
		m_core->m_shape.clear();
		for (int64 n = 0; n < (int64)shape.m_core->m_shape.size(); n++) {
			m_core->m_shape.push_back(shape.m_core->m_shape[n]);
		}
	}

	VShape copy() {
		VShape shape;
		for (int64 ax : m_core->m_shape) {
			shape.m_core->m_shape.push_back(ax);
		}
		return shape;
	}
	VShape transpose(VList axes) {
		VShape shape;
		if (size() != axes.size()) throw ValException(VERR_SHAPE_TRANSPOSE, __FILE__, __LINE__);
		int chk1 = 0, chk2 = 0;
		for (int64 n = 0; n < size(); n++) {
			shape.m_core->m_shape.push_back(m_core->m_shape[(int64)axes[n]]);
			chk1 |= 1 << n;
			chk2 |= 1 << (int64)axes[n];
		}
		if (chk1 != chk2) throw ValException(VERR_SHAPE_TRANSPOSE, __FILE__, __LINE__);
		return shape;
	}

	VShape replace_end(int64 axn) { 
		if (size() <= 0) throw ValException(VERR_SHAPE_REPLACE, __FILE__, __LINE__);
		VShape shape = copy();
		shape.m_core->m_shape[size() - 1] = axn;
		return shape;
	}
	VShape insert_head(int64 axn) {
		VShape shape;
		shape.m_core->m_shape.push_back(axn);
		for (int64 ax : m_core->m_shape) shape.m_core->m_shape.push_back(ax);
		return shape;
	}
	VShape insert_nth(int64 nth, int64 axn) {
		VShape shape;
		for (int64 n = 0; n < nth; n++) shape.m_core->m_shape.push_back(m_core->m_shape[n]);
		shape.m_core->m_shape.push_back(axn);
		for (int64 n = nth; n < size(); n++) shape.m_core->m_shape.push_back(m_core->m_shape[n]);
		return shape;
	}
	VShape append(int64 axn) {
		VShape shape = copy();
		shape.m_core->m_shape.push_back(axn);
		return shape;
	}
	VShape append(VShape tail) {
		VShape shape = copy();
		for (int64 n = 0; n < tail.size(); n++) shape.m_core->m_shape.push_back(tail[n]);
		return shape;
	}
	int64 tail_size(int64 axis) const {
		if (axis < 0) axis = kaxis_mod(axis, size());
		if (size() <= axis) return 1; // 20220508 dhyoon: 0에서 1로 수정함 
		int64 prod = 1;
		for (int64 n = axis; n < size(); n++) prod *= m_core->m_shape[n];
		return prod;
	}
	VShape cut_tail(int64 len) {
		if (size() < len) throw ValException(VERR_INPUT_LENGTH, __FILE__, __LINE__);
		VShape shape;
		for (int64 n = 0; n < size() - len; n++) shape.m_core->m_shape.push_back(m_core->m_shape[n]);
		return shape;
	}
	VShape tail(int64 axis) {
		int64 len = size() - axis;
		if (len < 0) throw ValException(VERR_SIZE_TENSOR, __FILE__, __LINE__);
		VShape shape;
		for (int64 n = 0; n < len; n++) shape.m_core->m_shape.push_back(m_core->m_shape[n + axis]);
		return shape;
	}
	VShape replace_tail(VShape oldTail, VShape newTail) {
		int64 len = size();
		int64 len1 = oldTail.size();
		int64 len2 = newTail.size();
		if (len < len1) throw ValException(VERR_INPUT_SHAPE, __FILE__, __LINE__);
		for (int64 n = len - len1, m = 0; n < len; n++, m++) {
			if (m_core->m_shape[n] != oldTail[m]) throw ValException(VERR_INPUT_SHAPE, __FILE__, __LINE__);
		}
		VShape shape;
		for (int64 n = 0; n < len - len1; n++) shape.m_core->m_shape.push_back(m_core->m_shape[n]);
		for (int64 n = 0; n < len2; n++) shape.m_core->m_shape.push_back(newTail[n]);
		return shape;
	}
	VShape replace_tail_by_size(VShape oldTail, VShape newTail) {
		int64 oldSize = oldTail.total_size();
		int64 prod = 1;
		int64 len1 = size();
		int64 len2 = newTail.size();
		while (len1 > 0) {
			if (prod >= oldSize) break;
			prod *= m_core->m_shape[--len1];
		}
		if (prod != oldSize) throw ValException(VERR_INPUT_SHAPE, __FILE__, __LINE__);
		VShape shape;
		for (int64 n = 0; n < len1; n++) shape.m_core->m_shape.push_back(m_core->m_shape[n]);
		for (int64 n = 0; n < len2; n++) shape.m_core->m_shape.push_back(newTail[n]);
		return shape;
	}
	VShape remove_tail_by_size(int64 tail_size) {
		int64 ax = size();
		while (--ax >= 0) {
			int64 axis_size = m_core->m_shape[ax];
			if (tail_size < axis_size) { throw ValException(VERR_INPUT_SHAPE, __FILE__, __LINE__); }
			else if (tail_size % axis_size != 0) { throw ValException(VERR_INPUT_SHAPE, __FILE__, __LINE__); }
			else {
				tail_size = tail_size / axis_size;
				if (tail_size == 1) break;
			}
		}
		VShape shape;
		for (int64 n = 0; n < ax; n++) shape.m_core->m_shape.push_back(m_core->m_shape[n]);
		return shape;
	}
	VShape remove_head_by_size(int64 head_size) {
		int64 len = size(), ax = -1;
		while (ax++ < len - 1) {
			int64 axis_size = m_core->m_shape[ax];
			if (head_size < axis_size) { throw ValException(VERR_INPUT_SHAPE, __FILE__, __LINE__); }
			else if (head_size % axis_size != 0) { throw ValException(VERR_INPUT_SHAPE, __FILE__, __LINE__); }
			else {
				head_size = head_size / axis_size;
				if (head_size == 1) break;
			}
		}
		VShape shape;
		for (int64 n = ax + 1; n < len; n++) shape.m_core->m_shape.push_back(m_core->m_shape[n]);
		return shape;
	}
	VShape remove_head(int64 cnt = 1) {
		int64 len = size() - cnt;
		if (len < 0) throw ValException(VERR_INPUT_LENGTH, __FILE__, __LINE__);
		VShape shape;
		for (int64 n = 0; n < len; n++) shape.m_core->m_shape.push_back(m_core->m_shape[n+cnt]);
		return shape;
	}
	VShape remove_end(int64 cnt = 1) {
		int64 len = size() - cnt;
		if (len < 0) throw ValException(VERR_INPUT_LENGTH, __FILE__, __LINE__);
		VShape shape;
		for (int64 n = 0; n < len; n++) shape.m_core->m_shape.push_back(m_core->m_shape[n]);
		return shape;
	}
	VShape remove_nth(int64 axis) {
		int64 len = size();
		if (axis < 0) axis = kaxis_mod(axis, size());
		if (axis < 0 || axis >= len) throw ValException(VERR_OUT_OF_RANGE, __FILE__, __LINE__);
		VShape shape;
		if (len == 1) {
			shape.m_core->m_shape.push_back(1);
		}
		else {
			for (int64 n = 0; n < axis; n++) shape.m_core->m_shape.push_back(m_core->m_shape[n]);
			for (int64 n = axis + 1; n < len; n++) shape.m_core->m_shape.push_back(m_core->m_shape[n]);
		}
		return shape;
	}
	VShape replace_nth(int64 axis, int64 count) {
		int64 len = size();
		if (axis < 0) axis = kaxis_mod(axis, size());
		if (axis >= len) throw ValException(VERR_OUT_OF_RANGE, __FILE__, __LINE__);
		VShape shape = copy();
		shape[axis] = count;
		return shape;
	}

	VShape replace_head(int64 count) {
		int64 len = size();
		if (len <= 0) throw ValException(VERR_SIZE_TENSOR, __FILE__, __LINE__);
		VShape shape = copy();
		shape[0] = count;
		return shape;
	}

	VShape resolve_plcaeholder(int64 to_size) {
		int64 len = size();
		int64 hidden = to_size / -total_size();
		if (len <= 0) throw ValException(VERR_SIZE_PLACEHOLDER, __FILE__, __LINE__);
		VShape shape = copy();

		for (int64 n = 0; n < len; n++) {
			if (shape[n] < -1) throw ValException(VERR_SHAPE_PLACEHOLDER, __FILE__, __LINE__);
			if (shape[n] > -1) continue;
			if (hidden <= 0)  throw ValException(VERR_SIZE_PLACEHOLDER, __FILE__, __LINE__);
			shape[n] = hidden;
			hidden = 0;
		}
		return shape;
	}

	int64& operator [](int64 ax) {
		if (size() == 0) {
			throw ValException(VERR_INDEXING_ON_NULL_SHAPE, __FILE__, __LINE__);
		}
		return m_core->m_shape[kaxis_mod(ax, size())];
	}

	int64 operator [](int64 ax) const {
		if (size() == 0) {
			throw ValException(VERR_INDEXING_ON_NULL_SHAPE, __FILE__, __LINE__);
		}
		return m_core->m_shape[kaxis_mod(ax, size())];
	}

	bool operator ==(const VShape& shape) const {
		int64 sz = size();
		if (sz != shape.size()) return false;
		for (int64 n = 0; n < sz; n++) if (m_core->m_shape[n] != shape[n]) return false;
		return true;
	}
	bool operator !=(const VShape& shape) const {
		int64 sz = size();
		if (sz != shape.size()) return true;
		for (int64 n = 0; n < sz; n++) {
			int64 n1 = m_core->m_shape[n];
			int64 n2 = shape[n];
			if (m_core->m_shape[n] != shape[n]) return true;
		}
		return false;
	}

	string desc() const {
		string desc, delimeter = "[";
		if (size() == 0) desc = "[";
		for (int64 n = 0; n < size(); n++) { desc += delimeter; delimeter = ",";  desc += std::to_string(m_core->m_shape[n]); }
		return desc + "]";
	}

	string batch_desc(bool bBatchBound, int64 nTimestepId) const {
		string desc = "[";
		string delimeter = "";
		int64 nFrom = 0;

		if (bBatchBound) desc += delimeter + "N", delimeter = ",", nFrom++;
		if (nTimestepId == 1) desc += delimeter + "T", delimeter = ",", nFrom++;	// 주의: 복합 키는 반영되어 있지 않음
		else if (nTimestepId > 1) desc += delimeter + "T" + std::to_string(nTimestepId), delimeter = ",", nFrom++;	// 주의: 복합 키는 반영되어 있지 않음
		for (int64 n = nFrom; n < size(); n++) { desc += delimeter; delimeter = ",";  desc += std::to_string(m_core->m_shape[n]); }
		return desc + "]";
	}

protected:
	VShapeCore* m_core;
	friend class VValue;
};

/*
class VShapeCore : public VObjCore {
	std::vector<int64> m_shape;

	friend class VShape;
	friend class VValue;
	friend class VValueCore;

	VShapeCore() : VObjCore(VObjType::shape) {}
	~VShapeCore() {}
	VShapeCore* clone() { return (VShapeCore*)clone_core(); }
};

class VShape {
public:
	VShape() { m_core = NULL; } // new VShapeCore(); }
	VShape(const VShape& src) { m_core = src.m_core->clone(); }
	VShape(VShapeCore* core) { m_core = core->clone(); }
	VShape(int64 dim, int64* ax_size) { m_core = new VShapeCore(); for (int64 n = 0; n < dim; n++) m_core->m_shape.push_back(ax_size[n]); }
	VShape(std::initializer_list<int64> list) { m_core = new VShapeCore(); for (int64 ax : list) m_core->m_shape.push_back(ax); }
	~VShape() { m_core->destroy(); }

	VShape& operator =(const VShape& src) {
		if (&src != this && m_core != src.m_core) {
			m_core->destroy();
			m_core = src.m_core->clone();
		}
		return *this;
	}

	//VShapeCore* detach() { m_core->m_nRefCnt++; return m_core; }

	int64 size() const { return m_core ? (int64)m_core->m_shape.size() : 0; }
	int64 total_size() const { if (size() == 0) return 0; int64 prod = 1; for (int64 ax : m_core->m_shape) prod *= ax; return prod; }
	int64 seed_size() const { int64 prod = 1; if (m_core) { for (int64 ax : m_core->m_shape) prod *= ax; } return prod; }
	VShape copy() {
		VShape shape;
		if (m_core) {
			shape.m_core = new VShapeCore();
			for (int64 ax : m_core->m_shape) {
				shape.m_core->m_shape.push_back(ax);
			}
		}
		return shape;
	}
	VShape replace_end(int64 axn) { if (size() <= 0) throw ValException(VERR_REPLACE_REQUEST_ON_EMPTY_SHAPE, __FILE__, __LINE__); VShape shape = copy(); shape.m_core->m_shape[size() - 1] = axn; return shape; }
	VShape insert_head(int64 axn) { VShape shape; if (m_core == NULL) shape.m_core = new VShapeCore();  shape.m_core->m_shape.push_back(axn); for (int64 ax : m_core->m_shape) shape.m_core->m_shape.push_back(ax); return shape; }
	VShape append(int64 axn) { VShape shape = copy(); if (m_core == NULL) shape.m_core = new VShapeCore();  shape.m_core->m_shape.push_back(axn); return shape; }
	VShape append(VShape tail) { VShape shape = copy(); if (m_core == NULL) shape.m_core = new VShapeCore();  for (int64 n = 0; n < tail.size(); n++) shape.m_core->m_shape.push_back(tail[n]); return shape; }
	int64 tail_size(int64 axis) const {
		if (axis < 0) axis = kaxis_mod(axis, size());
		if (size() <= axis) return 1; // 20220508 dhyoon: 0에서 1로 수정함 
		int64 prod = 1;
		for (int64 n = axis; n < size(); n++) prod *= m_core->m_shape[n];
		return prod;
	}
	VShape cut_tail(int64 len) {
		if (size() < len) throw ValException(VERR_INPUT_SHAPE_MISMATCH, __FILE__, __LINE__);
		VShape shape({});;
		for (int64 n = 0; n < size() - len; n++) shape.m_core->m_shape.push_back(m_core->m_shape[n]);
		return shape;
	}
	VShape tail(int64 axis) {
		int64 len = size() - axis;
		if (len < 0) throw ValException(VERR_INPUT_SHAPE_MISMATCH, __FILE__, __LINE__);
		VShape shape({});
		for (int64 n = 0; n < len; n++) shape.m_core->m_shape.push_back(m_core->m_shape[n + axis]);
		return shape;
	}
	VShape replace_tail(VShape oldTail, VShape newTail) {
		int64 len = size(), len1 = oldTail.size(), len2 = newTail.size();
		if (len < len1) throw ValException(VERR_INPUT_SHAPE_MISMATCH, __FILE__, __LINE__);
		for (int64 n = len - len1, m = 0; n < len; n++, m++) {
			if (m_core->m_shape[n] != oldTail[m]) throw ValException(VERR_INPUT_SHAPE_MISMATCH, __FILE__, __LINE__);
		}
		VShape shape({});
		for (int64 n = 0; n < len - len1; n++) shape.m_core->m_shape.push_back(m_core->m_shape[n]);
		for (int64 n = 0; n < len2; n++) shape.m_core->m_shape.push_back(newTail[n]);
		return shape;
	}
	VShape replace_tail_by_size(VShape oldTail, VShape newTail) {
		int64 oldSize = oldTail.total_size(), prod = 1;
		int64 len1 = size(), len2 = newTail.size();
		while (len1 > 0) {
			if (prod >= oldSize) break;
			prod *= m_core->m_shape[--len1];
		}
		if (prod != oldSize) throw ValException(VERR_INPUT_SHAPE_MISMATCH, __FILE__, __LINE__);
		VShape shape({});
		for (int64 n = 0; n < len1; n++) shape.m_core->m_shape.push_back(m_core->m_shape[n]);
		for (int64 n = 0; n < len2; n++) shape.m_core->m_shape.push_back(newTail[n]);
		return shape;
	}
	VShape remove_tail_by_size(int64 tail_size) {
		int64 ax = size();
		while (--ax >= 0) {
			int64 axis_size = m_core->m_shape[ax];
			if (tail_size < axis_size) { throw ValException(VERR_BAD_SHAPE_FOR_REMOVE_TAIL, __FILE__, __LINE__); }
			else if (tail_size % axis_size != 0) { throw ValException(VERR_BAD_SHAPE_FOR_REMOVE_TAIL, __FILE__, __LINE__); }
			else {
				tail_size = tail_size / axis_size;
				if (tail_size == 1) break;
			}
		}
		VShape shape({});
		for (int64 n = 0; n < ax; n++) shape.m_core->m_shape.push_back(m_core->m_shape[n]);
		return shape;
	}
	VShape remove_head_by_size(int64 head_size) {
		int64 len = size(), ax = -1;
		while (ax++ < len - 1) {
			int64 axis_size = m_core->m_shape[ax];
			if (head_size < axis_size) { throw ValException(VERR_BAD_SHAPE_FOR_REMOVE_HEAD, __FILE__, __LINE__); }
			else if (head_size % axis_size != 0) { throw ValException(VERR_BAD_SHAPE_FOR_REMOVE_HEAD, __FILE__, __LINE__); }
			else {
				head_size = head_size / axis_size;
				if (head_size == 1) break;
			}
		}
		VShape shape({});
		for (int64 n = ax + 1; n < len; n++) shape.m_core->m_shape.push_back(m_core->m_shape[n]);
		return shape;
	}
	VShape remove_end(int64 cnt = 1) {
		int64 len = size() - cnt;
		if (len < 0) throw ValException(VERR_INPUT_SHAPE_MISMATCH, __FILE__, __LINE__);
		VShape shape({});
		for (int64 n = 0; n < len; n++) shape.m_core->m_shape.push_back(m_core->m_shape[n]);
		return shape;
	}
	VShape remove_nth(int64 axis) {
		int64 len = size();
		if (axis < 0) axis = kaxis_mod(axis, size());
		if (axis < 0 || axis >= len) throw ValException(VERR_INPUT_SHAPE_MISMATCH, __FILE__, __LINE__);
		VShape shape({});
		if (len == 1) {
			shape.m_core->m_shape.push_back(1);
		}
		else {
			for (int64 n = 0; n < axis; n++) shape.m_core->m_shape.push_back(m_core->m_shape[n]);
			for (int64 n = axis + 1; n < len; n++) shape.m_core->m_shape.push_back(m_core->m_shape[n]);
		}
		return shape;
	}
	VShape replace_nth(int64 axis, int64 count) {
		int64 len = size();
		if (axis < 0) axis = kaxis_mod(axis, size());
		if (axis >= len) throw ValException(VERR_INPUT_SHAPE_MISMATCH, __FILE__, __LINE__);
		VShape shape = copy();
		shape[axis] = count;
		return shape;
	}

	int64& operator [](int64 ax) { return m_core->m_shape[kaxis_mod(ax, size())]; }
	int64 operator [](int64 ax) const { return m_core->m_shape[kaxis_mod(ax, size())]; }

	bool operator ==(const VShape& shape) const { int64 sz = size(); if (sz != shape.size()) return false; for (int64 n = 0; n < sz; n++) if (m_core->m_shape[n] != shape[n]) return false; return true; }
	bool operator !=(const VShape& shape) const {
		int64 sz = size();
		if (sz != shape.size()) return true;
		for (int64 n = 0; n < sz; n++) {
			int64 n1 = m_core->m_shape[n];
			int64 n2 = shape[n];
			if (m_core->m_shape[n] != shape[n]) return true;
		}
		return false;
	}

	string desc() const {
		string desc, delimeter = "[";
		if (size() == 0) desc = "[0";
		for (int64 n = 0; n < size(); n++) { desc += delimeter; delimeter = ",";  desc += std::to_string(m_core->m_shape[n]); }
		return desc + "]";
	}

	string batch_desc(bool bBatchBound, int64 nTimestepId) const {
		string desc = "[";
		string delimeter = "";
		int64 nFrom = 0;

		if (bBatchBound) desc += delimeter + "N", delimeter = ",", nFrom++;
		if (nTimestepId == 1) desc += delimeter + "T", delimeter = ",", nFrom++;	// 주의: 복합 키는 반영되어 있지 않음
		else if (nTimestepId > 1) desc += delimeter + "T" + std::to_string(nTimestepId), delimeter = ",", nFrom++;	// 주의: 복합 키는 반영되어 있지 않음
		for (int64 n = nFrom; n < size(); n++) { desc += delimeter; delimeter = ",";  desc += std::to_string(m_core->m_shape[n]); }
		return desc + "]";
	}

protected:
	VShapeCore* m_core;
	friend class VValue;
};
*/