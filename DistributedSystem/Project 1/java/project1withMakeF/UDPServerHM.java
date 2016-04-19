// UDPServerHM.java : a simple UDP server program.
import java.net.*;
import java.io.*;
import java.util.HashMap;
import java.util.Date;
import java.text.SimpleDateFormat;

public class UDPServerHM {
	protected DatagramSocket aSocket;
	protected HashMapKV hashmapKV;

	public void setHMKV(HashMapKV hashmapKV) {
		this.hashmapKV = hashmapKV;
	}
	public void setSocket(DatagramSocket aSocket) {
		this.aSocket = aSocket;
	}

	// method to receive and send message
	public void receiveSend(DatagramPacket request) {
		try{
			aSocket.receive(request);
			long curtime = System.currentTimeMillis();
			SimpleDateFormat sdf = new SimpleDateFormat("MMM dd,yyyy HH:mm:ss.SSS");    
			Date resultdate = new Date(curtime);
			String requestTime = sdf.format(resultdate);
			byte[] bytes = request.getData();
			String decodedRequest = new String(bytes,"UTF-8");
			System.out.println(decodedRequest);
			// parse request and output is the reply
			String decodedReply = parseRequest(decodedRequest);
			// call updateServerLog method to update the log
			updateServerLog(decodedRequest,requestTime,decodedReply);
			byte[] encodedReply = decodedReply.getBytes();
			DatagramPacket reply = new DatagramPacket(encodedReply,encodedReply.length,
			request.getAddress(),request.getPort());
			aSocket.send(reply);
		}
		catch (IOException e) {
			System.out.println("IO: " + e.getMessage());
		}
		
	}

	// method to generate / update server's log
	public void updateServerLog(String request, String requestTime, String reply) {
		File f = new File("serverlog.txt");
		long curtime = System.currentTimeMillis();
		SimpleDateFormat sdf = new SimpleDateFormat("MMM dd,yyyy HH:mm:ss.SSS");    
		Date resultdate = new Date(curtime);
		String replyTime = sdf.format(resultdate);
		if(f.exists() && !f.isDirectory()) { 
			try {
				PrintWriter writelog = new PrintWriter("serverlog.txt", "UTF-8");
				writelog.println("Request: " + request + "\t" + requestTime);
				writelog.println("Reply: " + reply + "\t" + replyTime);
				System.out.println("Request: " + request + "\t" + requestTime);
				System.out.println("Reply: " + reply + "\t" + replyTime);
				writelog.close();
    			System.out.println("Server log is updated as serverlog.txt");
			}
			catch (IOException ioe)
			{
    			System.err.println("Error: " + ioe.getMessage());
			}
		}
		else {
			try {
				PrintWriter writelog = new PrintWriter("serverlog.txt", "UTF-8");
				writelog.println("Request: " + request + "\t" + requestTime);
				writelog.println("Reply: " + reply + "\t" + replyTime);
				System.out.println("Request: " + request + "\t" + requestTime);
				System.out.println("Reply: " + reply + "\t" + replyTime);
				writelog.close();
				System.out.println("Server log is generated as serverlog.txt");
			}
			catch (IOException ioe){
				System.err.println("Error: " + ioe.getMessage());
			}
		}
	}

	// method to parse the decoded request and return an appropriate string as 
	// reply value.
	public String parseRequest(String decodedRequest) {
		String output;
		String[] elements = decodedRequest.split(" ");
		if(elements.length <= 2) {
			String func = elements[0];
			switch(func) {
				case "PUT":
					String pair = elements[1];
					if(pair.contains(":") && pair.length() >= 3) {
						String[] kv = pair.split(":");
						Object key = kv[0];
						Object value = kv[1];
						hashmapKV.put(key,value);
						output = "Key = "+key+", Value = "+value+" pair is added";
					}
					else {
						output = "Error: Incorrect Format; Should be <PUT key:value>";
					}
				break;
				case "DELETE":
					String key = elements[1];
					if(!key.contains(":") && !key.contains(" ")) {
						try {
							hashmapKV.delete(key);
							output = "Key "+key+" and its value are deleted from set";
						}
						catch (CustomException e) {
							StringWriter sw = new StringWriter();
							e.printStackTrace(new PrintWriter(sw));
							output = sw.toString().replace("CustomException: ","");
							System.out.println(e);
						}
					}
					else {
						output = "Error: only Key is required for DELETE function";
					}
				break;
				case "GET":
					key = elements[1];
					try {
						Object value = hashmapKV.get(key);
						output = "the value is "+value;
					}
					catch (CustomException e) {
						output = "key does not exist in out set";
					}
				break;
				default:
					output = "Error: Function is not found. Avaliable functions are PUT,DELETE and GET";		
			}
		}
		else {
			output = "Error: Input command should be formatted as <PUT key:value> or <DELETE key> or <GET key>";
		}
		return output;
	}
	

	public static void main(String args[]) {
		if(args.length < 1) {
			System.out.println("Usage: java UDPServerHM <Port Number>");
			System.exit(1);
		}
		try {
			DatagramSocket aSocket = null;
			int socket_no = Integer.valueOf(args[0]).intValue();
			aSocket = new DatagramSocket(socket_no);
			UDPServerHM udpServerHM = new UDPServerHM();
			udpServerHM.setSocket(aSocket);
			udpServerHM.setHMKV(new PlainHashMapKV());
			byte[] buffer = new byte[1000];
			while(true) {
				DatagramPacket request = new DatagramPacket(buffer,buffer.length);
				udpServerHM.receiveSend(request);
			}
		}
		catch (SocketException e) {
			System.out.println("Socket: " + e.getMessage());
		}
	}
}