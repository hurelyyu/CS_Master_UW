

import java.util.HashMap;

public class myHashMap {
	private HashMap<String, String> storageHashMap;
	
	public myHashMap() {
		storageHashMap = new HashMap<String, String>();
	}
	//if successfully operated, return integer 0
	public int put (String key, String value) {
		storageHashMap.put(key,value);
		return 0;
	}
	
	//if successfully operated, return integer 0
	public int delete (String key) {
		storageHashMap.remove(key);
		return 0;
	}

	//if successfully operated, return the value
	public String get (String key) {
		String requestedValue = storageHashMap.get(key);
		return requestedValue;
	}
	
	public String requestHandler(String request, String ip, int port) {
		String result = "Avaliable operation: put/<key>/<value>, get/<key>, delete/<key>.";
		int returnNum = 0;
		
		if (request == null) {
			myLog.error("Received malformed request from " + ip + ": " + port);
			result = "Request from " + ip + ": " + port + " corrupted.";
			return result;
		}
		
		String[] requestSplitted = request.split("/");	
		//Allow case-insensitive input
		String requestedOperation = requestSplitted[0].toLowerCase().trim();
		String key = requestSplitted[1].trim();
		
		if (requestedOperation.equals("put")) {
			String value = requestSplitted[2].trim();
			returnNum = put(key, value);
			if (returnNum == 0) {
				result = "PUT: ("+key+","+value+") Successfully operated.";
				myLog.normal("Request from " + ip + ": " + port +" succesfully executed : PUT (" + key + ", " + value + ")");
			} else {
				result = "Error: PUT operation failed. ";
				myLog.error("unknown error or received PUT request with invalid KEY and VALUE from " + ip + ": " + port);
			}
		} else if (requestedOperation.equals("delete")) {
			returnNum = delete(key);
			if (returnNum == 0) {
				result = "DELETE: \""+key+"\" Successfully operated.";
				myLog.normal("Request from " + ip + ": " + port +" succesfully executed : DELETE "+ key);
			} else {
				result = "Error: DELETE operation failed. ";
				myLog.error("unknown error or received DELETE request with a invalid KEY from " + ip + ": " + port);			
			}
		} else if (requestedOperation.equals("get")) {
			try {
				result = "Requested VALUE: " + get(key);				
				myLog.normal("Request from " + ip + ": " + port +" succesfully executed : GET VALUE of KEY " + key);
			} catch (Exception e) {
				result = "Error: GET operation failed. ";
				myLog.error("unknown error or received GET request with a invalid KEY from " + ip + ": " + port);
			}
		} else {
			myLog.error("Received unknown request: " + request);
		}
		return result;
	}
	
}
