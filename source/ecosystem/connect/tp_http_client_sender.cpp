#include "../connect/tp_http_client_sender.h"
#include "../connect/tp_nn_server.h"
#include "../connect/tp_api_conn.h"
#include "../utils/tp_json_parser.h"
#include "../utils/tp_utils.h"
#include "../utils/tp_exception.h"

NNRestClientBase::NNRestClientBase(string server_url) {
	try {
		utility::string_t clientAddr;
		clientAddr.assign(server_url.begin(), server_url.end());

		web::http::client::http_client_config clientCfg;
		clientCfg.set_timeout(utility::seconds(100));

		m_pClient = new web::http::client::http_client(clientAddr, clientCfg);
	}
	catch (...) {
		RestfulDebug(__FUNCTION__, __FILE__, __LINE__);
		TP_THROW(VERR_CREATE_RESTFUL_CLIENT);
	}
}

NNRestClientBase::~NNRestClientBase() {
}

VDict NNRestClientBase::execTransaction(string mainMenu, string subMenu, VDict requestArgs, void* pData, int64 nByteSize, bool bUpload) {
	VDict response;
	web::http::status_code statusCode;

	try {
		web::http::client::http_client* pRefClient = m_pClient;

		string menu = "/" + mainMenu + "/" + subMenu;
		utility::string_t url_menu(menu.begin(), menu.end());

		web::http::http_request req(web::http::methods::GET);
		web::uri_builder builder(url_menu);

		if (pData != NULL) {
			if (bUpload) {
				req.set_method(web::http::methods::POST);

				unsigned char* pChar = (unsigned char*)pData;
				std::vector<unsigned char> BodyData(pChar, pChar + nByteSize);

				req.set_body(BodyData);
			}
			else {
				req.set_method(web::http::methods::PUT);
			}
		}

		for (auto& it : requestArgs) {
			string sValue = it.second.desc();

			utility::string_t key(it.first.begin(), it.first.end());
			utility::string_t value(sValue.begin(), sValue.end());

			builder.append_query(key, value);
		}

		req.set_request_uri(builder.to_string());

		pRefClient->request(req).then([&builder, &response, &statusCode, &pData, &nByteSize, &bUpload](web::http::http_response resp) mutable \
				// lambda function : -->
        {
			statusCode = resp.status_code();
			
			if (statusCode == web::http::status_codes::OK) {
				if (pData && !bUpload) {
					resp.extract_vector().then([&pData, &nByteSize, &response](std::vector<unsigned char> BodyData)
						{
							memcpy(pData, BodyData.data(), nByteSize);
							response["result_code"] = 0;
						}).wait();
				}
				else {
					resp.extract_string(true).then([&response](utility::string_t BobyData)
						{
							string responseMessage = utility::conversions::to_utf8string(BobyData);
							response = JsonParser::ParseString(responseMessage);
							//std::wcout << BobyData << std::endl;
						}).wait();
				}
			}
		}).wait();
	}
	catch (...) {
		RestfulDebug(__FUNCTION__, __FILE__, __LINE__);
		TP_THROW(VERR_RESTFUL_CLIENT_REQUEST);
	}

	if (statusCode != web::http::status_codes::OK) {
		TP_THROW(VERR_RESTFUL_CLIENT_STATE);
	}

    return response;
}

NNRestClient::NNRestClient(string server_url, string client_url, ApiConn* pApiConn, VDict kwArgs) : NNRestClientBase(server_url) {
	try {
		m_connect(pApiConn, client_url, kwArgs);
	}
	catch (...) {
		RestfulDebug(__FUNCTION__, __FILE__, __LINE__);
		TP_THROW(VERR_RESTFUL_CLIENT_CONNECT);
	}
}

NNRestClient::~NNRestClient() {
}

VDict NNRestClient::execEngineExec(string apiName, VDict apiArgs, void* pData, int64 nByteSize, bool bUpload) {
	VDict requestArgs;

	requestArgs["api_name"] = apiName;
	requestArgs["api_arguments"] = apiArgs; // TpUtils::toJsonString(apiArgs);

	//print("Remotecall: %s", apiName.c_str());

	VDict responseArgs = execTransaction("engine", "exec", requestArgs, pData, nByteSize, bUpload);
	return responseArgs;
}

VDict NNRestClient::execTransaction(string mainMenu, string subMenu, VDict requestArgs, void* pData, int64 nByteSize, bool bUpload) {
	requestArgs["access_token"] = m_accessToken;
	VDict response = NNRestClientBase::execTransaction(mainMenu, subMenu, requestArgs, pData, nByteSize, bUpload);
	if ((int)response["result_code"] != 0) {
		TpException ex(response["result_code"], (VList)response["err_messages"]);
		TP_THROW3(TERR_EXEC_TRANSACTION, mainMenu + "/" + subMenu, ex);
	}
	return response;
}

void NNRestClient::m_connect(ApiConn* pApiConn, string client_url, VDict kwArgs) {
	string mainMenu = "conn";
	string subMenu = "connect";

	VDict requestArgs;

	requestArgs["response_type"] = "code";
	requestArgs["session_props"] = TpUtils::toJsonString(kwArgs);
	requestArgs["callback_url"] = client_url;
	requestArgs["callback_token"] = TpUtils::id_to_token((int64)pApiConn);
	
	string dump1 = requestArgs["session_props"];

	VDict responseArgs = execTransaction(mainMenu, subMenu, requestArgs);

	m_accessToken = (string)responseArgs["access_token"];
	m_expiresIn = responseArgs["expires_in"];
}

NNRestCbClient::NNRestCbClient(string url) : NNRestClientBase(url) {
}

NNRestCbClient::~NNRestCbClient() {
}
