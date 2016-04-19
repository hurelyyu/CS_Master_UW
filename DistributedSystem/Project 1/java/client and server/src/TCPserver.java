import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.IOException;
import java.io.InputStreamReader;
import java.io.OutputStreamWriter;
import java.io.PrintWriter;
import java.io.StringWriter;
import java.net.ServerSocket;
import java.net.Socket;
import java.net.SocketException;
import java.sql.Date;
import java.text.SimpleDateFormat;

public class TCPserver {
	//protected is a version of public restricted only to subclasses
	protected TCPserver tcpserver;
	protected Socket socket;
	protected HashmapYu hashmapyu;
	protected int portNumber;
	protected String request;
	protected String requesttime;
	
	public void setTCPserver(HashmapYu hashmapyu){
		this.hashmapyu = hashmapyu;
	}
	public void setSocket(Socket socket, int portNumber)
	{
		this.socket = socket;
		this.portNumber = portNumber;
	}
	private void setRequest(String request, String requesttime) {
		// TODO Auto-generated method stub
		this.request = request;
		this.requesttime = requesttime;
	}
	
	public void execute(){
		try{
			BufferedReader reader = new BufferedReader(new InputStreamReader(socket.getInputStream()));
			//read the message from client and parse the execution
			String line = reader.readLine();
			Long currenttime = System.currentTimeMillis();
			SimpleDateFormat sdf = new SimpleDateFormat("yyyy-MM-dd'T'HH:mm:ss.SSSXXX");
			
			//Date resulttime = new Date(currenttime);
			String date = sdf.format(new Date(currenttime));
			String reply;
			System.out.println(line);
			
			//request and request time setup
			setRequest(line, date);
			BufferedWriter writer = new BufferedWriter(new OutputStreamWriter(socket.getOutputStream()));
			//the predefined protocol for the hashmap is put/get/delete:key:value
			//element will be splitted to String[0], rest String[] after the first " "
			String[] elements = line.split(" ");
			//The body of a switch statement is known as a switch block. 
			//A statement in the switch block can be labeled with one or more case or default labels. 
			//The switch statement evaluates its expression
			//then executes all statements that follow the matching case label.
			switch(elements[0]){
			case "put":
				String sets = elements[1];
				if(sets.contains(":") && sets.length() >= 3 && elements.length == 2) {
					String[] kv = sets.split(":");
					Object key = kv[0];
					Object value = kv[1];
					hashmapyu.put(key, value);
					reply = "Key= " + key + "," + "Value= " + value + " sets are added";
					serverlog svlog = new serverlog(reply);
					writer.write(reply);
					writer.newLine();
				}
				else {
					reply = "input of key and value are in incorrect format key:value";
					serverlog svlog = new serverlog(reply);
					writer.write(reply);
					writer.newLine();
					
				}
				break;
			case "delete":
				if(elements.length == 2){
					String key = elements[1];
					if(!key.contains(":") && !key.contains(" ")){
						try{
							hashmapyu.delete(key);
							reply = "Key " + key + "and its value been deleted.";
							serverlog svlog = new serverlog(reply);
							writer.write(reply);
							writer.newLine();
						}catch(ClassCaseException e){
							StringWriter sw = new StringWriter();
						// prints the stack trace of the Exception to System.err.	
							e.printStackTrace(new PrintWriter(sw));
							String exceptiontoString = sw.toString().replace("ClassCaseException: ","");
							System.out.println(e);
							serverlog svlog = new serverlog(exceptiontoString);
							writer.write(exceptiontoString);
							writer.newLine();
						}
					}
					else {
						serverlog svlog = new serverlog("Key is requested for delete");
						writer.write("Key is requested for delete");
						writer.newLine();
					}
				}
				else {
					serverlog svlog = new serverlog("incorrect format; Should be <DELETE KEY>");
					writer.write("incorrect format; Should be <DELETE KEY>");
					writer.newLine();
				}
			break;
			case "get":
				if(elements.length == 2) {
					String key = elements[1];
					try {
						Object value = hashmapyu.get(key);
						serverlog svlog = new serverlog("the value is "+value);
						writer.write("the value is "+value);
						writer.newLine();
					}
					catch(ClassCaseException e){
						StringWriter sw = new StringWriter();
						e.printStackTrace(new PrintWriter(sw));
						String exceptionAsString = sw.toString().replace("ClassCastException: ","");
						System.out.println(e);
						serverlog svlog = new serverlog(exceptionAsString);
						writer.write(exceptionAsString);
						writer.newLine();
					}
				}
				else {
					reply = "incorrect format; Should be: <GET KEY>";
					serverlog svlog = new serverlog(reply);
					writer.write(reply);
					writer.newLine();
				}	
			break;
			default:
				reply = "Error: Function not found...";
				serverlog svlog = new serverlog(reply);
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
public static void main(String args[]){
	if(args.length < 1){
		System.out.println("TCPserver <port number>");
		//analogues to Error
		System.exit(1);
	}
	else{
		try {
			int portNumber = Integer.valueOf(args[0]);
			ServerSocket s = new ServerSocket(portNumber);
			TCPserver tcpserver = new TCPserver();
			tcpserver.setTCPserver(new ImplementHashmapYu()); 
			while(true){
				Socket socket = s.accept();
				tcpserver.setSocket(socket,portNumber);
				tcpserver.execute();
			}
		}
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

			