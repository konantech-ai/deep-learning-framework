#include "../connect/tp_nn_server.h"
#include "../connect/tp_api_conn.h"
#include "../connect/tp_http_client_sender.h"
#include "../objects/tp_nn.h"
#include "../objects/tp_tensor.h"
#include "../utils/tp_json_parser.h"
#include "../utils/tp_utils.h"
#include "../utils/tp_exception.h"

void RestfulDebug(char* FunctionName, char* FileName, int FileLine)
{
	try { std::rethrow_exception(std::current_exception()); }
	catch (std::exception& e) {
		print("C++ Exception / EFunction Name : %s() / %s / %d line%s", FunctionName, FileName, FileLine, e.what());
	}
	catch (...) {
		print("C++ Exception / EFunction Name : %s() / %s / %d line", FunctionName, FileName, FileLine);
	}
	return;
}

NNRestServerBase::NNRestServerBase() {
	m_pServer = NULL;
}

void NNRestServerBase::open(string addr) {
	try {
		utility::string_t serverAddr;
		serverAddr.assign(addr.begin(), addr.end());

		web::http::experimental::listener::http_listener_config serverCfg;
		serverCfg.set_timeout(utility::seconds(100));

		m_pServer = new web::http::experimental::listener::http_listener(serverAddr, serverCfg);

		m_serverAsyncGET();
		m_serverAsyncPUT();
		m_serverAsyncPOST();
		m_serverAsyncDELETE();
	}
	catch (...) { RestfulDebug(__FUNCTION__, __FILE__, __LINE__); }
}

NNRestServerBase::~NNRestServerBase() {
	try {
		if (m_pServer != NULL) {
			delete m_pServer;
			m_pServer = NULL;
		}
	}
	catch (...) { RestfulDebug(__FUNCTION__, __FILE__, __LINE__); }
}

void NNRestServerBase::openService() {
	try {
		m_pServer->open().wait();
	}
	catch (...) { RestfulDebug(__FUNCTION__, __FILE__, __LINE__); }
}

bool NNRestServerBase::m_serverAsyncGET(void) {
	try {
		web::http::experimental::listener::http_listener* pRefServer = m_pServer;
		
		pRefServer->support(web::http::methods::GET, [this](web::http::http_request req) mutable { // lambda function : -->
			RestTransactionItem item;

			TP_MEMO(VERR_UNDEFINED, "req로부터 보낸이 주소 찾아내 item.m_hostaddr에 지정할 것");
			item.m_hostaddr = "temp.com";

			const web::uri& url = req.request_uri();

			vector<utility::string_t> paths = url.split_path(url.path());
			map<utility::string_t, utility::string_t> queries = url.split_query(url.decode(url.query()));

			item.m_method = http_method::GET;

			for (auto& it : paths) {
				item.m_paths.push_back(utility::conversions::to_utf8string(it));
			}

			for (auto& it : queries) {
				string key = utility::conversions::to_utf8string(it.first);
				string sValue = utility::conversions::to_utf8string(it.second);
				
				item.m_queries[key] = JsonParser::ParseString(sValue);
			}

			item.m_statusCode = 0;
			web::http::status_code code = web::http::status_codes::OK;

			m_processRequest(&item);

			string response = TpUtils::toJsonString(item.m_response);
			
			utility::string_t message;
			message.assign(response.begin(), response.end());

			if (item.m_statusCode != 0) code = item.m_statusCode;

			req.reply(code, message);

			return false;
		}); // lambda function : <--

		return false;
	}
	catch (std::exception& e) { std::cout << e.what(); return true; }
}

bool NNRestServerBase::m_serverAsyncPUT(void) {
	try {
		web::http::experimental::listener::http_listener* pRefServer = m_pServer;

		pRefServer->support(web::http::methods::PUT, [this](web::http::http_request req) mutable { // lambda function : -->
			RestTransactionItem item;

			const web::uri& url = req.request_uri();

			vector<utility::string_t> paths = url.split_path(url.path());
			map<utility::string_t, utility::string_t> queries = url.split_query(url.decode(url.query()));

			item.m_method = http_method::PUT;

			for (auto& it : paths) {
				item.m_paths.push_back(utility::conversions::to_utf8string(it));
			}

			for (auto& it : queries) {
				string key = utility::conversions::to_utf8string(it.first);
				string sValue = utility::conversions::to_utf8string(it.second);

				item.m_queries[key] = JsonParser::ParseString(sValue);
			}

			item.m_statusCode = 0;

			m_processRequest(&item);

			web::http::http_response resp(web::http::status_codes::OK);

			unsigned char* pChar = (unsigned char*)item.m_pData;
			std::vector<unsigned char> BodyData(pChar, pChar + item.m_nByteSize);

			resp.set_body(BodyData);

			if (item.m_statusCode != 0) resp.set_status_code(item.m_statusCode);

			req.reply(resp);

			return false;
			}); // lambda function : <--

		return false;
	}
	catch (std::exception& e) { std::cout << e.what(); return true; }
}

bool NNRestServerBase::m_serverAsyncPOST(void) {
	try {
		web::http::experimental::listener::http_listener* pRefServer = m_pServer;

		pRefServer->support(web::http::methods::POST, [this](web::http::http_request req) mutable { // lambda function : -->
			RestTransactionItem item;

			const web::uri& url = req.request_uri();

			vector<utility::string_t> paths = url.split_path(url.path());
			map<utility::string_t, utility::string_t> queries = url.split_query(url.decode(url.query()));

			item.m_method = http_method::POST;

			for (auto& it : paths) {
				item.m_paths.push_back(utility::conversions::to_utf8string(it));
			}

			for (auto& it : queries) {
				string key = utility::conversions::to_utf8string(it.first);
				string sValue = utility::conversions::to_utf8string(it.second);

				item.m_queries[key] = JsonParser::ParseString(sValue);
			}

			item.m_statusCode = 0;
			web::http::status_code code = web::http::status_codes::OK;

			req.extract_vector().then([&item](std::vector<unsigned char> BodyData) {
				item.m_nByteSize = (int64)BodyData.size();
				item.m_pData = malloc(item.m_nByteSize);
					
				if (item.m_pData == NULL) TP_THROW2(VERR_HOSTMEM_ALLOC_FAILURE, to_string(item.m_nByteSize));

				memcpy(item.m_pData, BodyData.data(), item.m_nByteSize);
			}).wait();
				
			m_processRequest(&item);

			string response = TpUtils::toJsonString(item.m_response);

			utility::string_t message;
			message.assign(response.begin(), response.end());

			if (item.m_statusCode != 0) code = item.m_statusCode;

			req.reply(code, message);

			return false;
			}); // lambda function : <--

		return false;
	}
	catch (std::exception& e) { std::cout << e.what(); return true; }
}

bool NNRestServerBase::m_serverAsyncDELETE(void) {
	return true;
}

void NNRestServerBase::m_processRequest(RestTransactionItem* pItem) {
	pItem->m_statusCode = web::http::status_codes::NotFound;
}

NNRestCbServer::NNRestCbServer() : NNRestServerBase() {
}

NNRestCbServer::~NNRestCbServer() {
}

void NNRestCbServer::open(string addr) {
	NNRestServerBase::open(addr);
}

void NNRestCbServer::setClient(ApiConn* pApiConn, string callbackToken) {
	m_tokenMap[callbackToken] = TpUtils::id_to_token((int64)pApiConn);
}

void NNRestCbServer::m_processRequest(RestTransactionItem* pItem) {
	if (pItem->m_paths[0] == "engine") m_processEngineFamily(pItem);
	else if (pItem->m_paths[0] == "function") m_processFunctionFamily(pItem);
	else TP_THROW(VERR_NOT_IMPLEMENTED_YET);
}

void NNRestCbServer::m_processEngineFamily(RestTransactionItem* pItem) {
	if (pItem->m_paths[1] == "connect") {
		/*
		// 필요한 DB 작업, 로그 기록
		int64 sid = m_pDbConn->getNextId();
		int64 expiresIn = 100000;

		string accessToken = TpUtils::id_to_token(sid);

		VHSession hSession = NULL;
		string dump = pItem->m_queries["session_props"];
		VDict kwArgs = JsonParser::ParseString(pItem->m_queries["session_props"]);
		VDictWrapper wrapper(kwArgs);
		V_Session_open(&hSession, wrapper.detach());

		VDict record;

		record["uid"] = 0;
		record["sid"] = sid;
		record["ssid"] = 0;
		record["hsession"] = hSession;
		record["callback_url"] = pItem->m_queries["callback_url"];;
		record["callback_token"] = pItem->m_queries["callback_token"];;
		record["hostip"] = "www.temp.com";
		record["created"] = (int)time(NULL);
		record["last_access"] = (int)time(NULL);
		record["data"] = VDict();

		m_pDbConn->Insert("session", record);

		//m_handleRefCntMaps[hSession] = {};

		pItem->m_response["access_token"] = accessToken;
		pItem->m_response["expires_in"] = expiresIn;
		*/
		TP_THROW(VERR_NOT_IMPLEMENTED_YET);
	}
	else TP_THROW(VERR_NOT_IMPLEMENTED_YET);
}

void NNRestCbServer::m_processFunctionFamily(RestTransactionItem* pItem) {
	if (pItem->m_paths[1] == "forward") {
		string callback_token = pItem->m_queries["callback_token"];

		if (m_tokenMap.find(callback_token) == m_tokenMap.end()) {
			// 현재 callback_token 전달 과정에 착오가 있어서 m_tokenMap 키 대신 값에 callback_token이 들어 있음
			//TP_THROW(VERR_UNDEFINED);
		}

		void* pHandlerAux = (void*)(int64)pItem->m_queries["inst_info"];

		VHFunction hFunction = pItem->m_queries["function_handle"];
		int nInst = pItem->m_queries["function_inst"];
		VList opndHandles = pItem->m_queries["tensors"];
		VDict opArgs = pItem->m_queries["arguments"];

		VList tensorHandles = ApiConn::funcRemoteCbForwardHandler(pHandlerAux, hFunction, nInst, opndHandles, opArgs);

		pItem->m_response["tensorHandles"] = tensorHandles;
	}
	else if (pItem->m_paths[1] == "backward") {
		string callback_token = pItem->m_queries["callback_token"];

		if (m_tokenMap.find(callback_token) == m_tokenMap.end()) {
			// 현재 callback_token 전달 과정에 착오가 있어서 m_tokenMap 키 대신 값에 callback_token이 들어 있음
			//TP_THROW(VERR_UNDEFINED);
		}

		void* pHandlerAux = (void*)(int64)pItem->m_queries["inst_info"];

		VHFunction hFunction = pItem->m_queries["function_handle"];
		VList gradHandles = pItem->m_queries["gradients"];
		VList opndHandles = pItem->m_queries["tensors"];
		VDict opArgs = pItem->m_queries["arguments"];

		int nInst = pItem->m_queries["function_inst"];
		int nth = pItem->m_queries["nth"];

		VList tensorHandles = ApiConn::funcRemoteCbBackwardHandler(pHandlerAux, hFunction, nInst, gradHandles, opndHandles, opArgs, nth);

		pItem->m_response["tensorHandles"] = tensorHandles;
	}
	else TP_THROW(VERR_NOT_IMPLEMENTED_YET);
}
