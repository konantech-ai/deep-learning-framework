#pragma once

#include "../utils/tp_common.h"
#include "../objects/tp_nn.h"

class TpStream {
public:
	virtual ~TpStream();
	
	bool isOpened();

	void setOption(string key, VValue value);
    VValue getOption(string key);

protected:
    TpStream(ENN nn);

	ENN m_nn;
    FILE* m_fin;
    FILE* m_fout;
    VDict m_options;
};

class TpStreamIn : public TpStream {
public:
    TpStreamIn(ENN nn, string filename, bool bThrow);
	virtual ~TpStreamIn() {}

	bool load_bool();
	int load_int();
	int64 load_int64();
	VShape load_shape();
	float load_float();
	string load_string();
	void load_data(void* data, int64 size);
	VList load_list();
	VDict load_dict();
	VMap load_map();
	VIntList load_intlist();
	VStrList load_strlist();
	VHTensor load_tensor_handle();
	VValue load_value();
	ETensor load_tensor();
	ETensorDict load_tensordict();

	static bool load_bool(FILE* fin);
	static int load_int(FILE* fin);
	static int64 load_int64(FILE* fin);
	static VShape load_shape(FILE* fin);
	static float load_float(FILE* fin);
	static string load_string(FILE* fin);
	static void load_data(FILE* fin, void* data, int64 size);
	static VList load_list(FILE* fin);
	static VDict load_dict(FILE* fin);
	static VMap load_map(FILE* fin);
	static VIntList load_intlist(FILE* fin);
	static VStrList load_strlist(FILE* fin);
	static VHTensor load_tensor_handle(FILE* fin);
	static VValue load_value(FILE* fin);
	static ETensor load_tensor(FILE* fin);
	static ETensorDict load_tensordict(FILE* fin);

protected:
	FILE* m_getFin();
};

class TpStreamOut : public TpStream {
public:
	TpStreamOut(ENN nn, string filename);
	virtual ~TpStreamOut() {}

	void save_shape(VShape shape);
	void save_string(string dat);
	void save_bool(bool dat);
	void save_int(int dat);
	void save_int64(int64 dat);
	void save_float(float dat);
	void save_data(void* data, int64 size);
	void save_list(VList list);
	void save_dict(VDict dict);
	void save_map(VMap map);
	void save_intlist(VIntList list);
	void save_strlist(VStrList list);
	void save_tensor_handle(VObjCore* pObject);
	void save_value(VValue value);
	void save_tensor(ETensor tensor);
	void save_tensordict(ETensorDict tensors);

protected:
	FILE* m_getFout();
};
