import java.net.*;
import java.io.*;
import java.util.Date;
import java.util.HashMap;
import java.text.SimpleDateFormat;

public class TCPServerHM {
	protected HashMapKV hashmapKV;
	protected Socket socket;
	protected String request;
	protected int portNumber;
	protected String requestTime;

	public void setHMKV(HashMapKV hashmapKV) {
		this.hashmapKV = hashmapKV;
	}
	public void setSocket(Socket socket, int portNumber) {
		this.socket = socket;
		this.portNumber = portNumber;
	}
	public void setRequest(String request,String requestTime) {
		this.request = request;
		this.requestTime = requestTime;
	}

	// method to keep track of all request and reply for Server log
	public void updateServerLog(String reply) {
		File f = new File("serverlog.txt");
		long curtime = System.currentTimeMillis();
		SimpleDateFormat sdf = new SimpleDateFormat("MMM dd,yyyy HH:mm:ss.SSS");    
		Date resultdate = new Date(curtime);
		String datetime = sdf.format(resultdate);
		if(f.exists() && !f.isDirectory()) { 
			try {
				FileWriter fstream = new FileWriter("serverlog.txt", true); //true tells to append data.
    			BufferedWriter out = new BufferedWriter(fstream);
    			out.write("\nPortNumber: "+portNumber);
    			out.write("\nRequest: " + request + "\t" + requestTime);
    			out.write("\nReply" + reply + "\t" + datetime);
    			System.out.println("Request: " + request + "\t" + requestTime);
    			System.out.println("Reply: " + reply + "\t" + datetime);
    			out.close();
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
				writelog.println("PortNumber: "+portNumber);
				writelog.println("Request: " + request + "\t" + requestTime);
				writelog.println("Reply" + reply + "\t" + datetime);
				System.out.println("Request: " + request + "\t" + requestTime);
				System.out.println("Reply" + reply + "\t" + datetime);
				writelog.close();
				System.out.println("Server log is generated as serverlog.txt");
			}
			catch (IOException ioe){
				System.err.println("Error: " + ioe.getMessage());
			}
		}
	}
	public void execute() {

		// BufferedReader reader = new BufferedReader(
		// new InputStreamReader(socket.getInputStream()));
		// read the message from client and parse the execution
		try {
			// read the message from client and parse the execution
			BufferedReader reader = new BufferedReader(
				new InputStreamReader(socket.getInputStream()));
			String line = reader.readLine();
			long curtime = System.currentTimeMillis();
			SimpleDateFormat sdf = new SimpleDateFormat("MMM dd,yyyy HH:mm:ss.SSS");    
			Date resultdate = new Date(curtime);
			String datetime = sdf.format(resultdate);
			String reply;
			System.out.println(line);
			// set up request and the time that request is formed
			setRequest(line,datetime);
			String[] elements = line.split(" ");
			BufferedWriter writer = new BufferedWriter(
				new OutputStreamWriter(socket.getOutputStream()));
			switch (elements[0]) {
				case "PUT":
					String pair = elements[1];
					if(pair.contains(":") && pair.length() >= 3 && elements.length==2) {
						String[] kv = pair.split(":");
						Object key = kv[0];
						Object value = kv[1];
						hashmapKV.put(key,value);
						reply = "Key = "+key+", "+"Value = "+value+" pair is added";
						//System.out.println(reply);
						updateServerLog(reply);
						writer.write(reply);
						writer.newLine();
					}
					else {
						reply = "Error: input key and value are in incorrect format.PUT KEY:VALUE";
						updateServerLog(reply);
						writer.write(reply);
						writer.newLine();
					}
				break;
				case "DELETE":
					if(elements.length == 2) {
						String key = elements[1];
						if(!key.contains(":")&&!key.contains(" ")) {
							try {
								hashmapKV.delete(key);
								reply = "Key "+key+", and its value are deleted";
								updateServerLog(reply);
								writer.write(reply);
								writer.newLine();
							}
							catch(CustomException e) {
								StringWriter sw = new StringWriter();
								e.printStackTrace(new PrintWriter(sw));
								String exceptionAsString = sw.toString().replace("CustomException: ","");
								System.out.println(e);
								updateServerLog(exceptionAsString);
								writer.write(exceptionAsString);
								writer.newLine();
							}
						}
						else {
							updateServerLog("Error: only key is required for the DELETE function");
							writer.write("Error: Only key is required for the DELETE function");
							writer.newLine();
						}
					}
					else {
						updateServerLog("Error: incorrect format; Should be <DELETE KEY>");
						writer.write("Error: incorrect format; Should be <DELETE KEY>");
						writer.newLine();
					}
				break;
				case "GET":
					if(elements.length == 2) {
						String key = elements[1];
						try {
							Object value = hashmapKV.get(key);
							updateServerLog("the value is "+value);
							writer.write("the value is "+value);
							writer.newLine();
						}
						catch(CustomException e){
							StringWriter sw = new StringWriter();
							e.printStackTrace(new PrintWriter(sw));
							String exceptionAsString = sw.toString().replace("CustomException: ","");
							System.out.println(e);
							updateServerLog(exceptionAsString);
							writer.write(exceptionAsString);
							writer.newLine();
						}
					}
					else {
						reply = "Error: incorrect format; Should be: <GET KEY>";
						updateServerLog(reply);
						writer.write(reply);
						writer.newLine();
					}	
				break;
				default:
					reply = "Error: Function not found...";
					updateServerLog(reply);
					writer.write(reply);
					writer.newLine();
			}
			writer.close();
		}
		catch (SocketException e) {
			System.out.println("Socket: " + e.getMessage());
		}
		catch (IOException e) {
			System.out.println("IO: " + e.getMessage());
		}

	}
	

	public static void main(String args[]) {
		if(args.length < 1) {
			System.out.println("Usage: java tcpServerHM <Port Number>");
			System.exit(1);
		}
		else {
			try {
				int portNumber = Integer.valueOf(args[0]);
				// register service on port portNumber and run on certain Hostname
				ServerSocket s = new ServerSocket(portNumber);
				// the server should be listen to the client all the time 
				// wait and accept the connection until some external signal
				TCPServerHM tcpServerHM = new TCPServerHM();
				tcpServerHM.setHMKV(new PlainHashMapKV());
				while(true) {
					Socket socket = s.accept();
					// get a communication stream associated with the socket
					//OutputStream s1out = s1.getOutputStream();
					//DataOutputStream dos = new DataOutputStream(s1out);
					tcpServerHM.setSocket(socket,portNumber);
					tcpServerHM.execute();
				}
				
			}
			// if the args[0] is not integer type then return invalid argument
			catch(NumberFormatException e){
				System.out.println("Invalid port number");
			}
			catch (SocketException e) {
				System.out.println("Socket: " + e.getMessage());
			}
			catch (IOException e) {
				System.out.println("IO: " + e.getMessage());
			}
		}	
	}
}