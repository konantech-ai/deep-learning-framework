#include "../utils/tp_common.h"
#include "../utils/tp_exception.h"

#if defined(V_DEBUG_OBJ_LEAK)
int VObjCore::ms_nDebugNth = -1; // 739; // -1; // 추적을 원하는 객체 번호를 지정하세요. 없으면 -1
int VObjCore::ms_nDebugDomain = 1; // 추적을 원하는 도메인 번호를 지정하세요. 없으면 -1
int VObjCore::ms_nInstCount = 0;
int VObjCore::ms_nDomain = 1;
int VObjCore::ms_nth_nth = 0;
map<int, VObjCore*> VObjCore::ms_instAlive;
map<VObjCore*, int> VObjCore::ms_instDead;
#endif

int VObjCore::ms_nNextId = 0;
mutex VObjCore::ms_ref_mutex;

int VDataTypeSize(VDataType type) {
	switch (type) {
	case VDataType::int32:
	case VDataType::float32:
		return 4;
	case VDataType::int64:
		return 8;
	case VDataType::uint8:
	case VDataType::bool8:
		return 1;
	default:
		throw ValException(VERR_CONDITIONAL_STATEMENT, __FILE__, __LINE__);
	}
}

string VDataTypeName(VDataType type) {
	switch (type) {
	case VDataType::int32:
		return "int32";
	case VDataType::float32:
		return"float32";
	case VDataType::int64:
		return "int64";
	case VDataType::uint8:
		return "uint8";
	case VDataType::bool8:
		return "bool8";
	default:
		throw ValException(VERR_CONDITIONAL_STATEMENT, __FILE__, __LINE__);
	}
}

VValueCore::~VValueCore() {
	switch (m_type) {
	case VValueType::list:
	case VValueType::dict:
	case VValueType::map:
	case VValueType::object:
		m_value.m_pCore->destroy();
		m_value.m_pCore = NULL;
		break;
	case VValueType::shape:
		m_value.m_pShapeCore->destroy();
		m_value.m_pShapeCore = NULL;
		break;
	}
	//ms_debug_count--;
}

VValue::VValue(VList lValue) {
	m_core = new VValueCore();
	m_core->m_type = VValueType::list;
	m_core->m_value.m_pCore = lValue.m_core->clone();
}

VValue::VValue(VDict dValue) {
	m_core = new VValueCore();
	m_core->m_type = VValueType::dict;
	m_core->m_value.m_pCore = dValue.m_core->clone();
}

VValue::VValue(VMap mValue) {
	m_core = new VValueCore();
	m_core->m_type = VValueType::map;
	m_core->m_value.m_pCore = mValue.m_core->clone();
}

VValue::VValue(VShape sValue) {
	m_core = new VValueCore();
	m_core->m_type = VValueType::shape;
	m_core->m_value.m_pShapeCore = sValue.m_core->clone();
}

VValue::VValue(VObjCore* pValue) {
	m_core = new VValueCore();
	m_core->m_type = VValueType::object;
	//m_core->m_value.m_pCore = pValue; // ->clone_core();
	m_core->m_value.m_pCore = pValue->clone_core(); // 20220816: abalone 테스트 중 EFunction 객체 이상 삭제
}

VValue& VValue::operator =(VList lValue) {
	m_core->destroy();
	m_core = new VValueCore();
	m_core->m_type = VValueType::list;
	m_core->m_value.m_pCore = lValue.m_core->clone();
	return *this;
}

VValue& VValue::operator =(VDict dValue) {
	m_core->destroy();
	m_core = new VValueCore();
	m_core->m_type = VValueType::dict;
	m_core->m_value.m_pCore = dValue.m_core->clone();
	return *this;
}

VValue& VValue::operator =(VMap mValue) {
	m_core->destroy();
	m_core = new VValueCore();
	m_core->m_type = VValueType::map;
	m_core->m_value.m_pCore = mValue.m_core->clone();
	return *this;
}

VValue& VValue::operator =(VShape sValue) {
	m_core->destroy();
	m_core = new VValueCore();
	m_core->m_type = VValueType::shape;
	m_core->m_value.m_pShapeCore = sValue.m_core->clone();
	return *this;
}

VValue& VValue::operator =(VObjCore* pValue) {
	m_core->destroy();
	m_core = new VValueCore();
	m_core->m_type = VValueType::object;
	m_core->m_value.m_pCore = pValue->clone_core();
	return *this;
}

bool VValue::operator !=(const VValue& vValue) const {
	return !(*this == vValue);
}

bool VValue::operator ==(const VValue& vValue) const {
	VValueType vtype = vValue.type();

	switch (m_core->m_type) {
	case VValueType::kbool:
		return (bool)(*this) == (bool)vValue;
	case VValueType::int32:
		return (int)(*this) == (int)vValue;
	case VValueType::int64:
		return (int64)(*this) == (int64)vValue;
	case VValueType::float32:
		return (float)(*this) == (float)vValue;
	case VValueType::string:
		return (string)(*this) == (string)vValue;
	case VValueType::none:
		if (vtype == VValueType::none) return true;
		break;
	case VValueType::dict: // 내용이 아닌 포인터 수준의 비교? 내용으로 들어가야 할까?
		if (vtype != VValueType::dict) return false;
		return (VDictCore*)m_core->m_value.m_pCore == (VDictCore*)vValue.m_core->m_value.m_pCore;
	case VValueType::list: // 내용이 아닌 포인터 수준의 비교? 내용으로 들어가야 할까?
		if (vtype != VValueType::list) return false;
		return (VListCore*)m_core->m_value.m_pCore == (VListCore*)vValue.m_core->m_value.m_pCore;
	case VValueType::map: // 내용이 아닌 포인터 수준의 비교? 내용으로 들어가야 할까?
		if (vtype != VValueType::map) return false;
		return (VMapCore*)m_core->m_value.m_pCore == (VMapCore*)vValue.m_core->m_value.m_pCore;
	case VValueType::shape: // 내용이 아닌 포인터 수준의 비교? 내용으로 들어가야 할까?
		if (vtype != VValueType::shape) return false;
		return (VShapeCore*)m_core->m_value.m_pShapeCore == (VShapeCore*)vValue.m_core->m_value.m_pShapeCore;
	case VValueType::object:
		if (vtype != VValueType::object) return false;
		return m_core->m_value.m_pCore == vValue.m_core->m_value.m_pCore;
	}
	return false;
}

bool VValue::operator ==(const bool bValue) const {
	if (type() == VValueType::int32) return m_core->m_value.m_int32 == (int)bValue;
	if (type() == VValueType::int64) return m_core->m_value.m_int64 == (int64)bValue;
	if (type() == VValueType::float32) return m_core->m_value.m_float == (float)bValue;
	return false;
}

bool VValue::operator ==(int nValue) const {
	if (type() == VValueType::int32) return m_core->m_value.m_int32 == (int)nValue;
	if (type() == VValueType::int64) return m_core->m_value.m_int64 == (int64)nValue;
	if (type() == VValueType::float32) return m_core->m_value.m_float == (float)nValue;
	return false;
}

bool VValue::operator ==(int64 nValue) const {
	if (type() == VValueType::int32) return m_core->m_value.m_int32 == (int)nValue;
	if (type() == VValueType::int64) return m_core->m_value.m_int64 == (int64)nValue;
	if (type() == VValueType::float32) return m_core->m_value.m_float == (float)nValue;
	return false;
}

bool VValue::operator ==(string sValue) const {
	if (type() != VValueType::string) return false;
	return m_core->m_string == sValue;
}

bool VValue::operator ==(const char* sValue) const {
	if (type() != VValueType::string) return false;
	return m_core->m_string == (string)sValue;
}

bool VValue::operator ==(float fValue) const {
	if (type() == VValueType::int32) return (float)(m_core->m_value.m_int32) == fValue;
	if (type() == VValueType::int64) return (float)(m_core->m_value.m_int64) == fValue;
	if (type() == VValueType::float32) return m_core->m_value.m_float == fValue;
	return false;
}

bool VValue::operator ==(VList lValue) const {
	if (type() != VValueType::list) return false;
	return (VListCore*)m_core->m_value.m_pCore == lValue.m_core; // 내용이 아닌 포인터 수준의 비교? 내용으로 들어가야 할까?
}

bool VValue::operator ==(VDict dValue) const {
	if (type() != VValueType::dict) return false;
	return (VDictCore*)m_core->m_value.m_pCore == dValue.m_core; // 내용이 아닌 포인터 수준의 비교? 내용으로 들어가야 할까?
}

bool VValue::operator ==(VMap mValue) const {
	if (type() != VValueType::map) return false;
	return (VMapCore*)m_core->m_value.m_pCore == mValue.m_core; // 내용이 아닌 포인터 수준의 비교? 내용으로 들어가야 할까?
}

bool VValue::operator ==(VShape sValue) const {
	if (type() != VValueType::shape) return false;
	return (VShapeCore*)m_core->m_value.m_pShapeCore == sValue.m_core; // 내용이 아닌 포인터 수준의 비교? 내용으로 들어가야 할까?
}

bool VValue::operator ==(VObjCore* pValue) const {
	if (type() != VValueType::object) return false;
	return m_core->m_value.m_pCore == pValue;
}

VValue::operator int() const {
	if (type() == VValueType::int32) return (int)m_core->m_value.m_int32;
	if (type() == VValueType::int64) return (int)m_core->m_value.m_int64;
	if (type() == VValueType::float32) return (int)m_core->m_value.m_float;
	throw ValException(VERR_CONVERSION_VALUE, __FILE__, __LINE__);
	return 0;
}

VValue::operator int64() const {
	if (type() == VValueType::int32) return (int64)m_core->m_value.m_int32;
	if (type() == VValueType::int64) return (int64)m_core->m_value.m_int64;
	if (type() == VValueType::float32) return (int64)m_core->m_value.m_float;
	throw ValException(VERR_CONVERSION_VALUE, __FILE__, __LINE__);
	return 0;
}

VValue::operator int* () const {
	if (type() == VValueType::pint32) return m_core->m_value.m_pint32;
	throw ValException(VERR_CONVERSION_VALUE, __FILE__, __LINE__);
	return 0;
}

VValue::operator int64* () const {
	if (type() == VValueType::pint64) return m_core->m_value.m_pint64;
	throw ValException(VERR_CONVERSION_VALUE, __FILE__, __LINE__);
	return 0;
}

VValue::operator string() const {
	if (type() == VValueType::string) return m_core->m_string;
	throw ValException(VERR_CONVERSION_VALUE, __FILE__, __LINE__);
	return "";
}

VValue::operator bool() const {
	if (type() == VValueType::kbool) return m_core->m_value.m_bool;
	if (type() == VValueType::int32) return (bool)m_core->m_value.m_int32;
	if (type() == VValueType::int64) return (bool)m_core->m_value.m_int64;
	if (type() == VValueType::float32) return (bool)m_core->m_value.m_float;
	throw ValException(VERR_CONVERSION_VALUE, __FILE__, __LINE__);
	return 0;
}

VValue::operator float() const {
	if (type() == VValueType::int32) return (float)m_core->m_value.m_int32;
	if (type() == VValueType::int64) return (float)m_core->m_value.m_int64;
	if (type() == VValueType::float32) return m_core->m_value.m_float;
	throw ValException(VERR_CONVERSION_VALUE, __FILE__, __LINE__);
	return 0;
}

VValue::operator float* () const {
	if (type() == VValueType::pfloat) return m_core->m_value.m_pfloat;
	throw ValException(VERR_CONVERSION_VALUE, __FILE__, __LINE__);
	return NULL;
}

VValue::operator VList() {
	if (type() == VValueType::list) return VList((VListCore*)m_core->m_value.m_pCore);
	throw ValException(VERR_CONVERSION_VALUE, __FILE__, __LINE__);
	return VList();
}

VValue::operator VDict() {
	if (type() == VValueType::dict) return VDict((VDictCore*)m_core->m_value.m_pCore);
	throw ValException(VERR_CONVERSION_VALUE, __FILE__, __LINE__);
	return VDict();
}

VValue::operator VMap() {
	if (type() == VValueType::map) return VMap((VMapCore*)m_core->m_value.m_pCore);
	throw ValException(VERR_CONVERSION_VALUE, __FILE__, __LINE__);
	return VMap();
}

VValue::operator VShape() {
	if (type() == VValueType::shape) return VShape(m_core->m_value.m_pShapeCore);
	throw ValException(VERR_CONVERSION_VALUE, __FILE__, __LINE__);
	return VShape();
}

VValue::operator VObjCore*() const {
	if (type() == VValueType::object) return m_core->m_value.m_pCore;
	throw ValException(VERR_CONVERSION_VALUE, __FILE__, __LINE__);
	return 0;
}

static string encode_esc(string str) {
	size_t len = str.size();

	if (len >= 2 && str[0] == '\'' && str[len - 1] == '\'') {
		str = str.substr(1, len - 2);
	}

	if (str.find('\'') == std::string::npos && str.find('\\') == std::string::npos) return str;

	size_t pos = str.find('\\');
	while (pos != std::string::npos) {
		str = str.substr(0, pos - 1) + "\\" + str.substr(pos);
		pos = str.find('\\', pos + 2);
	}

	pos = str.find('\'');
	while (pos != std::string::npos) {
		str = str.substr(0, pos - 1) + "\\" + str.substr(pos);
		pos = str.find('\'', pos + 2);
	}

	return str;
}

static string list_desc(VListCore* pList) {
}

string VValue::desc() const {
	switch (type()) {
	case VValueType::none:
		return "None";
	case VValueType::kbool:
		return m_core->m_value.m_bool ? "True" : "False";
	case VValueType::int32:
		return std::to_string(m_core->m_value.m_int32);
	case VValueType::int64:
		return std::to_string(m_core->m_value.m_int64);
	case VValueType::float32:
		return std::to_string(m_core->m_value.m_float);
	case VValueType::string:
		return "'" + encode_esc(m_core->m_string) + "'";
	case VValueType::list:
		{
			VListCore* pList = (VListCore*)m_core->m_value.m_pCore;
			string desc, delimeter = "[";
			int64 size = pList->m_list.size();
			if (size == 0) desc = delimeter;
			for (int64 n = 0; n < size; n++) {
				desc += delimeter; delimeter = ",";
				desc += pList->m_list[n].desc();
			}
			return desc + "]";
		}
	case VValueType::dict:
		{
			VDictCore* pDict = (VDictCore*)m_core->m_value.m_pCore;
			std::map<string, VValue> dict = pDict->m_dict;
			string desc, delimeter = "{";
			if (dict.size() == 0) desc = delimeter;
			for (VDictIter it = dict.begin(); it != dict.end(); it++) {
				desc += delimeter; delimeter = ",";
				desc += "'" + it->first + "':" + it->second.desc();
			}
			return desc + "}";
		}
	case VValueType::map:
	{
		VMapCore* pMap = (VMapCore*)m_core->m_value.m_pCore;
		std::map<int, VValue> map = pMap->m_map;
		string desc, delimeter = "{";
		if (map.size() == 0) desc = delimeter;
		for (VMapIter it = map.begin(); it != map.end(); it++) {
			desc += delimeter; delimeter = ",";
			desc += std::to_string(it->first) + ":" + it->second.desc();
		}
		return desc + "}";
	}
	case VValueType::shape:
	{
		VShapeCore* pShape = m_core->m_value.m_pShapeCore;
		string desc, delimeter = "<";
		int64 size = pShape->m_shape.size();
		if (size == 0) desc = delimeter;
		for (int64 n = 0; n < size; n++) { desc += delimeter; delimeter = ",";  desc += std::to_string(pShape->m_shape[n]); }
		return desc + ">";
	}
	case VValueType::object:
		//return m_core->m_value.m_pCore->desc();
		return "#" + std::to_string((int64)m_core->m_value.m_pCore);
	}

	throw ValException(VERR_CONDITIONAL_STATEMENT, __FILE__, __LINE__);

	return "";
}

bool VList::find_string(string sValue) const {
	for (auto& it : m_core->m_list) {
		if (it.type() == VValueType::string && (string)it == sValue) return true;
	}
	return false;
}

bool VDict::operator ==(VDict dValue) const {
	return !(*this != dValue);
}

bool VDict::operator !=(VDict dValue) const {
	if (size() != dValue.size()) return true;

	for (auto& it : *this) {
		if (dValue.find(it.first) == dValue.end()) return true;
		if (it.second != dValue[it.first]) return true;
	}

	return false;
}