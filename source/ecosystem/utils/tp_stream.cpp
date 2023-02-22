#include "../utils/tp_stream.h"
#include "../connect/tp_api_conn.h"
#include "../objects/tp_tensor.h"
#include "../objects/tp_tensor_core.h"
#include "../utils/tp_utils.h"
#include "../utils/tp_exception.h"

TpStream::TpStream(ENN nn) {
	m_nn = nn;
	m_fin = NULL;
	m_fout = NULL;
}

TpStream::~TpStream() {
	if (m_fin) fclose(m_fin);
	if (m_fout) fclose(m_fout);
}

bool TpStream::isOpened() {
	return m_fin != NULL;
}

void TpStream::setOption(string key, VValue value) {
	m_options[key] = value;
}

VValue TpStream::getOption(string key) {
	return m_options[key];
}

TpStreamIn::TpStreamIn(ENN nn, string filename, bool bThrow) : TpStream(nn) {
	m_fin = TpUtils::fopen(filename.c_str(), "rb", bThrow);
	if (m_fin == NULL) {
		if (bThrow) TP_THROW(VERR_FILE_OPEN);
	}
}

TpStreamOut::TpStreamOut(ENN nn, string filename) : TpStream(nn) {
	m_fout = TpUtils::fopen(filename.c_str(), "wb");
	if (m_fout == NULL) TP_THROW(VERR_FILE_OPEN);
}

FILE* TpStreamOut::m_getFout() {
	if (m_fout == NULL) TP_THROW(VERR_INVALID_FILE);
	return m_fout;
}

void TpStreamOut::save_shape(VShape shape) {
	int len = (int)shape.size();
	save_int(len);

	for (int i = 0; i < len; i++) {
		save_int((int)shape[i]);
	}
}

void TpStreamOut::save_int(int dat) {
	FILE* fid = m_getFout();
	if (fwrite(&dat, sizeof(int), 1, fid) != 1) TP_THROW(VERR_FILE_WRITE);
}

void TpStreamOut::save_bool(bool dat) {
	FILE* fid = m_getFout();
	if (fwrite(&dat, sizeof(bool), 1, fid) != 1) TP_THROW(VERR_FILE_WRITE);
}

void TpStreamOut::save_int64(int64 dat) {
	FILE* fid = m_getFout();
	if (fwrite(&dat, sizeof(int64), 1, fid) != 1) TP_THROW(VERR_FILE_WRITE);
}

void TpStreamOut::save_float(float dat) {
	FILE* fid = m_getFout();
	if (fwrite(&dat, sizeof(float), 1, fid) != 1) TP_THROW(VERR_FILE_WRITE);
}

void TpStreamOut::save_string(string dat) {
	int len = (int)dat.size();
	save_int(len);

	FILE* fid = m_getFout();
	if (fwrite(dat.c_str(), sizeof(char), len, fid) != len) TP_THROW(VERR_FILE_WRITE);
}

void TpStreamOut::save_data(void* data, int64 size) {
	FILE* fid = m_getFout();
	if (fwrite(data, sizeof(char), size, fid) != size) TP_THROW(VERR_FILE_WRITE);
}

void TpStreamOut::save_list(VList list) {
	save_int((int)list.size());
	for (auto& it : list) {
		save_value(it);
	}
}

void TpStreamOut::save_dict(VDict dict) {
	save_int((int)dict.size());
	for (auto& it : dict) {
		save_string(it.first);
		save_value(it.second);
	}
}

void TpStreamOut::save_map(VMap map) {
	save_int((int)map.size());
	for (auto& it : map) {
		save_int(it.first);
		save_value(it.second);
	}
}

void TpStreamOut::save_strlist(VStrList list) {
	save_int((int)list.size());
	for (auto& it : list) {
		save_string(it);
	}
}

void TpStreamOut::save_intlist(VIntList list) {
	save_int((int)list.size());
	for (auto& it : list) {
		save_int(it);
	}
}

void TpStreamOut::save_tensor(ETensor tensor) {
	tensor.save(this);
}

void TpStreamOut::save_tensor_handle(VObjCore* pObject) {
	VObjType objType = pObject->getType();
	
	if (objType != VObjType::Tensor) {
		TP_THROW(VERR_NOT_IMPLEMENTED_YET);
	}

	save_int((int)objType);

	VHTensor hTensor = (VHTensor)pObject;

	VShape shape;
	VDataType dataType;
	int nDevice;

	m_nn.getApiConn()->Tensor_getFeature(hTensor, &shape, &dataType, &nDevice, __FILE__, __LINE__);

	save_shape(shape);
	save_string(TpUtils::to_string(dataType));
	save_int(nDevice);

	bool bSaveData = TpUtils::seekDict(m_options, "save_tensor_contents", false);

	if (shape.size() == 0) bSaveData = false;

	save_bool(bSaveData);

	if (bSaveData) {
		int64 size = shape.total_size() * TpUtils::byte_size(dataType);
		void* pData = malloc(size);
		if (pData == NULL) TP_THROW2(VERR_HOSTMEM_ALLOC_FAILURE, to_string(size));
		m_nn.getApiConn()->Tensor_downloadData(hTensor, pData, size, __FILE__, __LINE__);
		save_data(pData, size);
		free(pData);
	}
}

void TpStreamOut::save_tensordict(ETensorDict dict) {
	save_int((int)dict.size());
	for (auto& it : dict) {
		save_string(it.first);
		save_tensor(it.second);
	}
}

void TpStreamOut::save_value(VValue value) {
	save_int((int)value.type());

	switch (value.type()) {
	case VValueType::none:
		break;
	case VValueType::kbool:
		save_bool((bool)value);
		break;
	case VValueType::int32:
		save_int((int)value);
		break;
	case VValueType::int64:
		save_int64((int64)value);
		break;
	case VValueType::float32:
		save_float((float)value);
		break;
	case VValueType::pint32:
		TP_THROW(VERR_NOT_IMPLEMENTED_YET);
		break;
	case VValueType::pint64:
		TP_THROW(VERR_NOT_IMPLEMENTED_YET);
		break;
	case VValueType::pfloat:
		TP_THROW(VERR_NOT_IMPLEMENTED_YET);
		break;
	case VValueType::string:
		save_string((string)value);
		break;
	case VValueType::list:
		save_list((VList)value);
		break;
	case VValueType::dict:
		save_dict((VDict)value);
		break;
	case VValueType::map:
		save_map((VMap)value);
		break;
	case VValueType::shape:
	{
		VShape temp = value;
		save_shape(temp);
		break;
	}
	case VValueType::object:
		save_tensor_handle((VObjCore*)value);
		break;
	default:
		TP_THROW(VERR_NOT_IMPLEMENTED_YET);
		break;
	}
}

FILE* TpStreamIn::m_getFin() {
	if (m_fin == NULL) TP_THROW(VERR_INVALID_FILE);
	return m_fin;
}

bool TpStreamIn::load_bool() {
	FILE* fid = m_getFin();
	bool dat;
	if (fread(&dat, sizeof(bool), 1, fid) != 1) TP_THROW(VERR_FILE_READ);
	return dat;
}

bool TpStreamIn::load_bool(FILE* fin) {
	bool dat;
	if (fread(&dat, sizeof(bool), 1, fin) != 1) TP_THROW(VERR_FILE_READ);
	return dat;
}

int TpStreamIn::load_int() {
	FILE* fid = m_getFin();
	int dat;
	if (fread(&dat, sizeof(int), 1, fid) != 1) TP_THROW(VERR_FILE_READ);
	return dat;
}

int TpStreamIn::load_int(FILE* fin) {
	int dat;
	if (fread(&dat, sizeof(int), 1, fin) != 1) TP_THROW(VERR_FILE_READ);
	return dat;
}

int64 TpStreamIn::load_int64() {
	FILE* fid = m_getFin();
	int64 dat;
	if (fread(&dat, sizeof(int64), 1, fid) != 1) TP_THROW(VERR_FILE_READ);
	return dat;
}

int64 TpStreamIn::load_int64(FILE* fin) {
	int64 dat;
	if (fread(&dat, sizeof(int64), 1, fin) != 1) TP_THROW(VERR_FILE_READ);
	return dat;
}

float TpStreamIn::load_float() {
	FILE* fid = m_getFin();
	float dat;
	if (fread(&dat, sizeof(float), 1, fid) != 1) TP_THROW(VERR_FILE_READ);
	return dat;
}

float TpStreamIn::load_float(FILE* fin) {
	float dat;
	if (fread(&dat, sizeof(float), 1, fin) != 1) TP_THROW(VERR_FILE_READ);
	return dat;
}

VShape TpStreamIn::load_shape() {
	VShape shape;

	int len = load_int();

	for (int i = 0; i < len; i++) {
		int axis_size = load_int();
		shape = shape.append(axis_size);
	}

	return shape;
}

VShape TpStreamIn::load_shape(FILE* fin) {
	VShape shape;

	int len = load_int(fin);

	for (int i = 0; i < len; i++) {
		int axis_size = load_int(fin);
		shape = shape.append(axis_size);
	}

	return shape;
}

string TpStreamIn::load_string() {
	FILE* fid = m_getFin();
	int len = load_int();
	char* buffer = new char[len + 1];
	if (buffer == NULL) TP_THROW2(VERR_HOSTMEM_ALLOC_FAILURE, to_string(len + 1));
	if (fread(buffer, sizeof(char), len, fid) != len) TP_THROW(VERR_FILE_READ);
	string result = string(buffer, len);
	delete[] buffer;
	return result;
}

string TpStreamIn::load_string(FILE* fin) {
	int len = load_int(fin);
	char* buffer = new char[len + 1];
	if (buffer == NULL) TP_THROW2(VERR_HOSTMEM_ALLOC_FAILURE, to_string(len + 1));
	if (fread(buffer, sizeof(char), len, fin) != len) TP_THROW(VERR_FILE_READ);
	string result = string(buffer, len);
	delete[] buffer;
	return result;
}

void TpStreamIn::load_data(void* data, int64 size) {
	FILE* fid = m_getFin();
	if (fread(data, sizeof(char), size, fid) != size) TP_THROW(VERR_FILE_READ);
}

void TpStreamIn::load_data(FILE* fin, void* data, int64 size) {
	if (fread(data, sizeof(char), size, fin) != size) TP_THROW(VERR_FILE_READ);
}

VList TpStreamIn::load_list() {
	VList list;

	int cnt = load_int();

	for (int n = 0; n < cnt; n++) {
		VValue value = load_value();
		list.push_back(value);
	}

	return list;
}

VList TpStreamIn::load_list(FILE* fin) {
	VList list;

	int cnt = load_int(fin);

	for (int n = 0; n < cnt; n++) {
		VValue value = load_value(fin);
		list.push_back(value);
	}

	return list;
}

VDict TpStreamIn::load_dict() {
	VDict dict;

	int cnt = load_int();

	for (int n = 0; n < cnt; n++) {
		string key = load_string();
		VValue value = load_value();

		dict[key] = value;
	}

	return dict;
}

VDict TpStreamIn::load_dict(FILE* fin) {
	VDict dict;

	int cnt = load_int(fin);

	for (int n = 0; n < cnt; n++) {
		string key = load_string(fin);
		VValue value = load_value(fin);

		dict[key] = value;
	}

	return dict;
}

VMap TpStreamIn::load_map() {
	VMap map;

	int cnt = load_int();

	for (int n = 0; n < cnt; n++) {
		int key = load_int();
		VValue value = load_value();

		map[key] = value;
	}

	return map;
}

VMap TpStreamIn::load_map(FILE* fin) {
	VMap map;

	int cnt = load_int(fin);

	for (int n = 0; n < cnt; n++) {
		int key = load_int(fin);
		VValue value = load_value(fin);

		map[key] = value;
	}

	return map;
}

VStrList TpStreamIn::load_strlist() {
	VStrList list;

	int cnt = load_int();

	for (int n = 0; n < cnt; n++) {
		list.push_back(load_string());
	}

	return list;
}

VStrList TpStreamIn::load_strlist(FILE* fin) {
	VStrList list;

	int cnt = load_int(fin);

	for (int n = 0; n < cnt; n++) {
		list.push_back(load_string(fin));
	}

	return list;
}

VIntList TpStreamIn::load_intlist() {
	VIntList list;

	int cnt = load_int();

	for (int n = 0; n < cnt; n++) {
		list.push_back(load_int());
	}

	return list;
}

VIntList TpStreamIn::load_intlist(FILE* fin) {
	VIntList list;

	int cnt = load_int(fin);

	for (int n = 0; n < cnt; n++) {
		list.push_back(load_int(fin));
	}

	return list;
}

ETensorDict TpStreamIn::load_tensordict() {
	ETensorDict dict;

	int cnt = load_int();

	for (int n = 0; n < cnt; n++) {
		string key = load_string();
		ETensor tensor = load_tensor();

		dict[key] = tensor;
	}

	return dict;
}

ETensorDict TpStreamIn::load_tensordict(FILE* fin) {
	ETensorDict dict;

	int cnt = load_int(fin);

	for (int n = 0; n < cnt; n++) {
		string key = load_string(fin);
		ETensor tensor = load_tensor(fin);

		dict[key] = tensor;
	}

	return dict;
}

VHTensor TpStreamIn::load_tensor_handle() {
	VObjType objType = (VObjType)load_int();

	if (objType != VObjType::Tensor) {
		TP_THROW(VERR_NOT_IMPLEMENTED_YET);
	}

	VShape shape = load_shape();
	VDataType dataType = TpUtils::to_data_type(load_string());
	int nDevice = load_int();

	VHTensor hTensor = m_nn.getApiConn()->Tensor_create(__FILE__, __LINE__);
	
	m_nn.getApiConn()->Tensor_setFeature(hTensor, shape, dataType, nDevice, __FILE__, __LINE__);

	ETensor tensor(m_nn, shape, dataType);

	bool bSaveData = load_bool();

	if (bSaveData) {
		int64 size = shape.total_size() * TpUtils::byte_size(dataType);
		void* pData = malloc(size);
		if (pData == NULL) TP_THROW2(VERR_HOSTMEM_ALLOC_FAILURE, to_string(size));
		load_data(pData, size);
		m_nn.getApiConn()->Tensor_uploadData(hTensor, pData, size, __FILE__, __LINE__);
		free(pData);
	}

	return hTensor;
}

VHTensor TpStreamIn::load_tensor_handle(FILE* fin) {
	TP_THROW(VERR_NOT_IMPLEMENTED_YET);
}

ETensor TpStreamIn::load_tensor() {
	return ETensor::load(m_nn, this);
}

ETensor TpStreamIn::load_tensor(FILE* fin) {
	TP_THROW(VERR_INTERNAL_LOGIC);
}

VValue TpStreamIn::load_value() {
	VValueType type = (VValueType)load_int();

	switch (type) {
	case VValueType::none:
		break;
	case VValueType::kbool:
		return load_bool();
	case VValueType::int32:
		return load_int();
	case VValueType::int64:
		return load_int64();
	case VValueType::float32:
		return load_float();
	case VValueType::pint32:
		TP_THROW(VERR_NOT_IMPLEMENTED_YET);
		break;
	case VValueType::pint64:
		TP_THROW(VERR_NOT_IMPLEMENTED_YET);
		break;
	case VValueType::pfloat:
		TP_THROW(VERR_NOT_IMPLEMENTED_YET);
		break;
	case VValueType::string:
		return load_string();
	case VValueType::list:
		return load_list();
		break;
	case VValueType::dict:
		return load_dict();
		break;
	case VValueType::map:
		return load_map();
	case VValueType::shape:
		return load_shape();
		break;
	case VValueType::object:
		return load_tensor_handle();
		break;
	}

	TP_THROW(VERR_CONDITIONAL_STATEMENT);
}

VValue TpStreamIn::load_value(FILE* fin) {
	VValueType type = (VValueType)load_int(fin);

	switch (type) {
	case VValueType::none:
		break;
	case VValueType::kbool:
		return load_bool(fin);
	case VValueType::int32:
		return load_int(fin);
	case VValueType::int64:
		return load_int64(fin);
	case VValueType::float32:
		return load_float(fin);
	case VValueType::pint32:
		TP_THROW(VERR_NOT_IMPLEMENTED_YET);
		break;
	case VValueType::pint64:
		TP_THROW(VERR_NOT_IMPLEMENTED_YET);
		break;
	case VValueType::pfloat:
		TP_THROW(VERR_NOT_IMPLEMENTED_YET);
		break;
	case VValueType::string:
		return load_string(fin);
	case VValueType::list:
		return load_list(fin);
		break;
	case VValueType::dict:
		return load_dict(fin);
		break;
	case VValueType::map:
		return load_map(fin);
	case VValueType::shape:
		return load_shape(fin);
		break;
	case VValueType::object:
		return load_tensor_handle(fin);
		break;
	}

	TP_THROW(VERR_CONDITIONAL_STATEMENT);
}

