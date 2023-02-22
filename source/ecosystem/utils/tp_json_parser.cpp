#include "../utils/tp_json_parser.h"
#include "../utils/tp_utils.h"
#include "../utils/tp_exception.h"

JsonParser::JsonParser() {
}

JsonParser::~JsonParser() {
}

VValue JsonParser::ParseFile(string sFilePath) {
	JsonParser parser;
	return parser.parse_file(sFilePath);
}

VValue JsonParser::ParseFile(FILE* fid) {
	JsonParser parser;
	return parser.parse_file(fid);
}

VValue JsonParser::ParseString(string sJson) {
	JsonParser parser;
	StringLookAheader ahead(sJson);
	return parser.parse_json(ahead);
}

VValue JsonParser::parse_file(string sFilePath) {
	FileLookAheader aheader(sFilePath);
	VValue result = parse_json(aheader);
	return result;
}

VValue JsonParser::parse_jsonl_file(string sFilePath) {
	VList lines;
	FileLookAheader aheader(sFilePath);

	try {
		while (true) {
			VValue value = parse_json(aheader);
			if (value.is_none()) break;
			lines.push_back(value);
		}
	}
	catch (...) {
	}

	return lines;
}

VValue JsonParser::parse_file(FILE* fid) {
	FileLookAheader aheader(fid);
	return parse_json(aheader);
}

VValue JsonParser::parse_json(LookAheader& aheader) {
	int ch = aheader.look();
	
	if (ch == '[') return decode_list(aheader);
	else if (ch == '{') return decode_dict(aheader);
	else if ((ch >= '0' && ch <= '9') || ch == '-') return decode_number(aheader);
	else if (ch == '\'' || ch == '"') return decode_string(aheader);
	else if (ch == '#') {
		VHandle handle = decode_handle(aheader);
		VValue value = (int64)handle;
		return value;
	}
	else if (ch == '<' || ch == '(') return decode_shape(aheader, ch);
	else return decode_bool(aheader);
}

VList JsonParser::decode_list(LookAheader& aheader) {
	VList list;

	if (aheader.at_end()) return list;

	aheader.check('[');
	int ch = aheader.look();

	if (ch != ']') {
		while (true) {
			VValue value = parse_json(aheader);
			list.push_back(value);
			if (aheader.look() == ']') break;
			aheader.check(',');
			if (aheader.look() == ']') break;
			ch = aheader.look();
		}
	}

	aheader.check(']');
	return list;
}

VShape JsonParser::decode_shape(LookAheader& aheader, int left) {
	VShape shape;

	if (aheader.at_end()) return shape;

	aheader.check(left);
	int ch = aheader.look();
	int right = (left == '<') ? '>' : ')';

	if (ch != right) {
		while (true) {
			VValue value = parse_json(aheader);
			shape = shape.append((int64)value);
			if (aheader.look() == right) break;
			aheader.check(',');
			if (aheader.look() == right) break;
			ch = aheader.look();
		}
	}

	aheader.check(right);
	return shape;
}

VDict JsonParser::decode_dict(LookAheader& aheader) {
	VDict dict;

	if (aheader.at_end()) return dict;

	aheader.check('{');
	int ch = aheader.look();

	if (ch != '}') {
		while (true) {
			string key = decode_string(aheader);
			aheader.check(':');
			VValue value = parse_json(aheader);
			dict[key] = value;
			if (aheader.look() == '}') break;
			aheader.check(',');
			if (aheader.look() == '}') break;
			ch = aheader.look();
		}
	}

	aheader.check('}');
	return dict;
}

VValue JsonParser::decode_number(LookAheader& aheader) {
	int64 value = 0, sign = 1;

	if (aheader.pass('-')) sign = -1;

	while (aheader.look() >= '0' && aheader.look() <= '9') {
		value = value * 10 + (aheader.get() - '0');
	}

	if (aheader.pass('.')) {
		float fvalue = (float)value, unit = (float)0.1;
		while (aheader.look() >= '0' && aheader.look() <= '9') {
			fvalue = fvalue + (float)(aheader.get() - '0') * unit;
			unit *= (float)0.1;
		}
		return (float)sign * fvalue;
	}
	else if (aheader.look() == 'e' || aheader.look() == 'E') {
		TP_THROW(VERR_JSON_PARSING);
	}

	return sign * value;
}

VHandle JsonParser::decode_handle(LookAheader& aheader) {
	aheader.check('#');

	int64 value = 0;

	while (aheader.look() >= '0' && aheader.look() <= '9') {
		value = value * 10 + (aheader.get() - '0');
	}

	return (VHandle)value;
}

string JsonParser::decode_string(LookAheader& aheader) {
	int quote = aheader.get();
	if (quote != '\'' && quote != '"') {
		aheader.report_err("missing quote for string");
		TP_THROW(VERR_JSON_PARSING);
	}
	return aheader.substr(quote);
	/*
	const char* from = ++rest;
	while (*rest++ != quote) {
		assert(*rest != 0);
	}
	return string(from, rest-from-1);
	*/
}

bool JsonParser::decode_bool(LookAheader& aheader) {
	if (aheader.next("True")) return true;
	else if (aheader.next("False")) return false;
	TP_THROW(VERR_JSON_PARSING);
	return false;
}

int LookAheader::look() {
	m_skip_space();
	return m_buffer[m_begin];
}

int LookAheader::get() {
	m_skip_space();
	return m_buffer[m_begin++];
}

void LookAheader::report_err(string msg) {
	size_t left_from = (m_begin > 10) ? m_begin - 10 : 0;
	size_t left_size = (m_begin > 10) ? 10 : m_begin;

	size_t len = m_buffer.size();

	string left = m_buffer.substr(left_from, left_size);

	size_t right_from = m_begin + 1;
	size_t right_size = ((size_t)m_begin < len - 10) ? 10 : len - m_begin - 1;

	string right = m_buffer.substr(right_from, right_size);

	int ch = m_buffer[m_begin];

	TP_THROW(VERR_JSON_PARSING); // , msg, right);

	//clogger.Print("json parsing error: %s%s...%c...%s", msg.c_str(), left.c_str(), ch, right.c_str());
}

void LookAheader::check(int ch) {
	int read = get();
	if (read != ch) {
		char buffer[128];
		report_err(buffer);
	}
}

bool LookAheader::at_end() {
	m_skip_space();
	return m_at_end();
}

bool LookAheader::pass(int ch) {
	if (look() != ch) return false;
	m_begin++;
	return true;
}

string LookAheader::substr(int ch) {
	int pos = m_begin;
	int read;

	while ((read = m_buffer[pos++]) != ch) {
		if (read == '\\') pos++;
		if (pos >= m_end) {
			pos = pos - m_begin;
			if (!m_read_buffer()) TP_THROW(VERR_JSON_PARSING);;
		}
	}

	string result = m_buffer.substr(m_begin, pos - m_begin - 1);

	m_begin = pos;

	size_t esc_pos = result.find('\\');

	while (esc_pos != string::npos) {
		result = result.substr(0, esc_pos) + result.substr(esc_pos + 1);
		esc_pos = result.find('\\', esc_pos + 1);
	}

	return result;
}

bool LookAheader::next(string str) {
	int length = (int)str.length();
	m_skip_space();
	while (m_end - m_begin < length) {
		if (!m_read_buffer()) TP_THROW(VERR_JSON_PARSING);;
	}
	if (m_buffer.substr(m_begin, length) == str) {
		m_begin += length;
		return true;
	}
	return false;
}

void LookAheader::m_skip_space() {
	while (true) {
		if (m_begin >= m_end) {
			if (!m_read_buffer()) break;
		}
		else {
			if (!isspace(m_buffer[m_begin])) break;
			m_begin++;
		}
	}
}

StringLookAheader::StringLookAheader(string exp) : LookAheader() {
	m_buffer = exp;
	m_begin = 0;
	m_end = (int)m_buffer.length();
}

StringLookAheader::~StringLookAheader() {
	while (m_begin < m_end && isspace(m_buffer[m_begin])) m_begin++;
	if (m_begin < m_end && m_buffer[m_begin] == 0) m_begin++;
	assert(m_begin == m_end);
}

bool StringLookAheader::m_read_buffer() {
	return false;
}

bool StringLookAheader::m_at_end() {
	return m_begin >= m_end;
};

FileLookAheader::FileLookAheader(string path) {
	m_fid = TpUtils::fopen(path.c_str(), "rt");
	m_needToClose = true;
	m_begin = 0;
	m_end = 0;
}

FileLookAheader::FileLookAheader(FILE* fid) {
	m_fid = fid;
	m_needToClose = false;
	m_begin = 0;
	m_end = 0;
}

FileLookAheader::~FileLookAheader() {
	if (m_needToClose) fclose(m_fid);
	else {
		int curr = ftell(m_fid);
		fseek(m_fid, m_begin - m_end, SEEK_CUR);
		int next = ftell(m_fid);
		int n = 0;
	}
}

bool FileLookAheader::m_read_buffer() {
	char buffer[10241];
	int nread = (int)fread(buffer, sizeof(char), 10240, m_fid);
	if (nread <= 0) {
		//TP_THROW(VERR_FAIL_ON_JSON_PARSING);
		return false;
	}
	buffer[nread] = 0;

	string read_str = (string)buffer;
	
	int read_size = (int)read_str.length();

	if (m_begin < m_end) {
		m_buffer = m_buffer.substr(m_begin, m_end - m_begin) + read_str;
	}
	else {
		m_buffer = read_str;
	}

	m_begin = 0;
	m_end = (int)m_buffer.length();

	if (read_size < nread) {
		fseek(m_fid, read_size - nread, SEEK_CUR);
	}
	
	return true;
}

bool FileLookAheader::m_at_end() {
	if (m_begin < m_end) return false;
	return feof(m_fid);
}

