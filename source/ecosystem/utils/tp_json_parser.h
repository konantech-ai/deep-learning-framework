#pragma once

#include "../utils/tp_common.h"

class LookAheader {
public:
	int look();
	int get();
	void check(int ch);	// 반드시 특정 문자 입력 받아야 함
	bool at_end();
	bool pass(int ch); // 지정 문자면 통과하면서 true, 아니면 false
	string substr(int ch);
	bool next(string str);
	void report_err(string msg);

protected:
	string m_buffer;
	int m_begin, m_end;

	void m_skip_space();

	virtual bool m_read_buffer() = 0;
	virtual bool m_at_end() = 0;
};

class FileLookAheader : public LookAheader {
public:
	FileLookAheader(string path);
	FileLookAheader(FILE* fid);
	virtual ~FileLookAheader();

protected:
	virtual bool m_read_buffer();
	virtual bool m_at_end();

	FILE* m_fid;
	bool m_needToClose;
	int m_pos;
};

class StringLookAheader : public LookAheader {
public:
	StringLookAheader(string exp);
	virtual ~StringLookAheader();

protected:
	virtual bool m_read_buffer();
	virtual bool m_at_end();
};

class JsonParser {
public:
	JsonParser();
	virtual ~JsonParser();

	static VValue ParseFile(string sFilePath);
	static VValue ParseFile(FILE* fid);
	static VValue ParseString(string sJson);

	VValue parse_file(string sFilePath);
	VValue parse_jsonl_file(string sFilePath);
	VValue parse_file(FILE* fid);
	VValue parse_json(LookAheader& aheader);

	VList decode_list(LookAheader& aheader);
	VDict decode_dict(LookAheader& aheader);
	VValue decode_number(LookAheader& aheader);
	string decode_string(LookAheader& aheader);
	bool decode_bool(LookAheader& aheader);
	VHandle decode_handle(LookAheader& aheader);
	VShape decode_shape(LookAheader& aheader, int left);
};

