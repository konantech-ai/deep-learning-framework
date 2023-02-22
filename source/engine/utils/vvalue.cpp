#include "../api/vcommon.h"
#include "../include/vvalue.h"
#include "../utils/vexception.h"

//#define TP_THROW(code) VP_THROW(code);

#if defined(V_DEBUG_OBJ_LEAK)
int VObjCore::ms_nDebugNth = -1; // 13419, 13520, -1; // 2542; // 추적을 원하는 객체 번호를 지정하세요. 없으면 -1
int VObjCore::ms_nDebugDomain = 0; // -1; // 추적을 원하는 도메인 번호를 지정하세요. 없으면 -1
int VObjCore::ms_nInstCount = 0;
int VObjCore::ms_nDomain = 0;
int VObjCore::ms_nth_nth = 0;
//set<int> VObjCore::ms_instIds;
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
		return 1;
	default:
		TP_THROW(VERR_NOT_IMPLEMENTED_YET);
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
		TP_THROW(VERR_NOT_IMPLEMENTED_YET);
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
	m_core->m_value.m_pCore = pValue; // ->clone_core();
}

VValue& VValue::operator =(VList lValue) {
	if (m_core->m_type != VValueType::list && m_core->m_value.m_pCore != lValue.m_core) {
		m_core->destroy();
		m_core = new VValueCore();
		m_core->m_type = VValueType::list;
		m_core->m_value.m_pCore = lValue.m_core->clone();
	}
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
	m_core->m_value.m_pCore = pValue; // ->clone_core();
	return *this;
}

bool VValue::operator ==(const VValue& vValue) const {
	VValueType vtype = vValue.type();

	switch (m_core->m_type) {
	case VValueType::int32:
		return (int)(*this) == (int)vValue;
	case VValueType::int64:
		return (int64)(*this) == (int64)vValue;
	case VValueType::float32:
		return (float)(*this) == (float)vValue;
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
	TP_THROW(VERR_CONVERSION_VALUE);
	return 0;
}

VValue::operator int64() const {
	if (type() == VValueType::int32) return (int64)m_core->m_value.m_int32;
	if (type() == VValueType::int64) return (int64)m_core->m_value.m_int64;
	if (type() == VValueType::float32) return (int64)m_core->m_value.m_float;
	TP_THROW(VERR_CONVERSION_VALUE);
	return 0;
}

VValue::operator int* () const {
	if (type() == VValueType::pint32) return m_core->m_value.m_pint32;
	TP_THROW(VERR_CONVERSION_VALUE);
	return 0;
}

VValue::operator int64* () const {
	if (type() == VValueType::pint64) return m_core->m_value.m_pint64;
	TP_THROW(VERR_CONVERSION_VALUE);
	return 0;
}

VValue::operator string() const {
	if (type() == VValueType::string) return m_core->m_string;
	TP_THROW(VERR_CONVERSION_VALUE);
	return "";
}

VValue::operator bool() const {
	if (type() == VValueType::kbool) return m_core->m_value.m_bool;
	if (type() == VValueType::int32) return (bool)m_core->m_value.m_int32;
	if (type() == VValueType::int64) return (bool)m_core->m_value.m_int64;
	if (type() == VValueType::float32) return (bool)m_core->m_value.m_float;
	TP_THROW(VERR_CONVERSION_VALUE);
	return 0;
}

VValue::operator float() const {
	if (type() == VValueType::int32) return (float)m_core->m_value.m_int32;
	if (type() == VValueType::int64) return (float)m_core->m_value.m_int64;
	if (type() == VValueType::float32) return m_core->m_value.m_float;
	TP_THROW(VERR_CONVERSION_VALUE);
	return 0;
}

VValue::operator float* () const {
	if (type() == VValueType::pfloat) return m_core->m_value.m_pfloat;
	TP_THROW(VERR_CONVERSION_VALUE);
	return NULL;
}

VValue::operator VList() {
	if (type() == VValueType::list) return VList((VListCore*)m_core->m_value.m_pCore);
	TP_THROW(VERR_CONVERSION_VALUE);
	return VList();
}

VValue::operator VDict() {
	if (type() == VValueType::dict) return VDict((VDictCore*)m_core->m_value.m_pCore);
	TP_THROW(VERR_CONVERSION_VALUE);
	return VDict();
}

VValue::operator VMap() {
	if (type() == VValueType::map) return VMap((VMapCore*)m_core->m_value.m_pCore);
	TP_THROW(VERR_CONVERSION_VALUE);
	return VMap();
}

VValue::operator VShape() {
	if (type() == VValueType::shape) return VShape(m_core->m_value.m_pShapeCore);
	if (type() == VValueType::list) return VShape((VList)*this);
	if(0) {
		printf("BP1: value not vshape: %s\n", this->desc().c_str());
		printf("BP1: value is_list: %d\n", this->is_list());
	}
	TP_THROW(VERR_CONVERSION_VALUE);
	return VShape();
}

VValue::operator VObjCore* () const {
	if (type() == VValueType::object) return m_core->m_value.m_pCore;
	else if (type() == VValueType::int64) return (VObjCore*)m_core->m_value.m_int64;
	TP_THROW(VERR_CONVERSION_VALUE);
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

	TP_THROW(VERR_INTERNAL_LOGIC);
	return "";
}

bool VList::find_string(string sValue) const {
	for (auto& it : m_core->m_list) {
		if (it.type() == VValueType::string && (string)it == sValue) return true;
	}
	return false;
}

/*
for (int nr = 0; nr < nrow; nr++) {
	for (int nc = 0; nc < ncol; nc++) {
		for (int nn = 0; nn < chns; nn++) {
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
					float xpixel = m_get_pixel(img, ny, nx, nn);
					ypixel += xpixel * (xt * xratio) * (yt * yratio);
				}
			}

			m_set_pixel(pImageBuf, ncol, chns, nr, nc, nn, ypixel);
		}
	}
}
*/