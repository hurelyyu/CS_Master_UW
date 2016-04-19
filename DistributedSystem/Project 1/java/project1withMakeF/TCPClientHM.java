import java.net.*;
import java.io.*;
import java.util.HashMap;
import java.util.Date;
import java.text.*;

public class TCPClientHM {
	Socket socket;

	// setup socket
	public void setSocket(Socket socket) {
		this.socket = socket;
	}
	// method to get the result from the server
	public void sendReceive(String sendMessage) {
		try {
			BufferedWriter writer = new BufferedWriter(
			new OutputStreamWriter(socket.getOutputStream()));
			writer.write(sendMessage);
			writer.newLine();
			writer.flush();
			// get the result from the server
			BufferedReader reader = new BufferedReader(
				new InputStreamReader(socket.getInputStream()));
			String result = reader.readLine();
			// for every excution or error will be recorded in client log
			updateClientLog(result);
			reader.close();
			writer.close();	
		}
		catch(Exception e) {
			System.out.println("Fail to Write or Read");
		}
	}

	// method to generate client log for every output
	public void updateClientLog(String output) {
		File f = new File("clientlog.txt");
		long curtime = System.currentTimeMillis();
		SimpleDateFormat sdf = new SimpleDateFormat("MMM dd,yyyy HH:mm:ss.SSS");    
		Date resultdate = new Date(curtime);
		String datetime = sdf.format(resultdate);
		if(f.exists() && !f.isDirectory()) { 
			try {
				FileWriter fstream = new FileWriter("clientlog.txt", true); //true tells to append data.
    			BufferedWriter out = new BufferedWriter(fstream);
    			out.write("\n" + output + "\t" + datetime);
    			System.out.println(output + "\t" + datetime);
    			out.close();
    			System.out.println("Client log is updated as clientlog.txt");
			}
			catch (IOException ioe)
			{
    			System.err.println("Error: " + ioe.getMessage());
			}
		}
		else {
			try {
				PrintWriter writelog = new PrintWriter("clientlog.txt", "UTF-8");
				writelog.println(output + "\t" + datetime);
				System.out.println(output + "\t" + datetime);
				writelog.close();
				System.out.println("Client log is generated as clientlog.txt");
			}
			catch (IOException ioe){
				System.err.println("Error: " + ioe.getMessage());
			}
		}
	}

	public static void main(String args[]) {
		if(args.length < 3) {
			System.out.println("Usage: java TCPClientHM <Hostname> <Port Number> <Function key:value>/<Function key>");
		}
		else {
			String hostname = args[0];
			try {
				int port = Integer.parseInt(args[1]);
				// create a socket
				// perform a simple operation PUT(1,3)
				Socket socket = new Socket();
				socket.connect(new InetSocketAddress(hostname,port),10000);
				socket.setSoTimeout(10000);
				String x = args[2];
				TCPClientHM tcpClientHM = new TCPClientHM();
				tcpClientHM.setSocket(socket);
				tcpClientHM.sendReceive(sendMessage);
			}
			catch (SocketException iioe)
			{
   				File f = new File("clientlog.txt");
				long curtime = System.currentTimeMillis();
				SimpleDateFormat sdf = new SimpleDateFormat("MMM dd,yyyy HH:mm:ss.SSS");    
				Date resultdate = new Date(curtime);
				String datetime = sdf.format(resultdate);
				System.err.println("Timed out to connect server at port: "+args[1]+" "+datetime);
				if(f.exists() && !f.isDirectory()) { 
					try {
						FileWriter fstream = new FileWriter("clientlog.txt", true); //true tells to append data.
    					BufferedWriter out = new BufferedWriter(fstream);
    					out.write("\nFailed to connect to the Server at port: "+args[1]+" at "+ datetime);
    					out.write("\nFailed to send: "+args[2]);
    					System.out.println("Client log is updated as clientlog.txt");
    					out.close();
					}
					catch (IOException ioe)
					{
    					System.err.println("Error: " + ioe.getMessage());
					}
				}
				else {
					try {
						PrintWriter writelog = new PrintWriter("clientlog.txt", "UTF-8");
						writelog.println("Failed to connect to the Server at port: "+args[1]+" at "+ datetime);
						writelog.println("Failed to send: "+args[2]);
						writelog.close();
						System.out.println("Client log is generated as clientlog.txt");
					}
					catch (IOException ioe){
						System.err.println("Error: " + ioe.getMessage());
					}
				}
			}
			catch (Exception e) {
				System.out.println("Fail to connect to the Server at port: "+args[1]);
				File f = new File("clientlog.txt");
				long curtime = System.currentTimeMillis();
				SimpleDateFormat sdf = new SimpleDateFormat("MMM dd,yyyy HH:mm:ss.SSS");    
				Date resultdate = new Date(curtime);
				String datetime = sdf.format(resultdate);
				if(f.exists() && !f.isDirectory()) { 
					try {
						FileWriter fstream = new FileWriter("clientlog.txt", true); //true tells to append data.
    					BufferedWriter out = new BufferedWriter(fstream);
    					out.write("\nFailed to connect to the Server at port: "+args[1]+" at "+ datetime);
    					out.write("\nFailed to send: "+args[2]);
    					System.out.println("Client log is updated as clientlog.txt");
    					out.close();
					}
					catch (IOException ioe)
					{
    					System.err.println("Error: " + ioe.getMessage());
					}
				}
				else {
					try {
						PrintWriter writelog = new PrintWriter("clientlog.txt", "UTF-8");
						writelog.println("Failed to connect to the Server at port: "+args[1]+" at "+ datetime);
						writelog.println("Failed to send: "+args[2]);
						writelog.close();
						System.out.println("Client log is generated as clientlog.txt");
					}
					catch (IOException ioe){
						System.err.println("Error: " + ioe.getMessage());
					}
				}
			}
		}

	}
}