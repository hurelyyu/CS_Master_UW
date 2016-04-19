

import java.io.*;
import java.net.*;

public class TCPServerMain {
	/*
	 * args[0] gives server's port
	 */
	public static void main(String args[]) {

		try {
			myLog.createLog("TCPServerLog.txt");
		} catch (IOException e) {
			System.out.println("Error: Log creation for TCP server failed.");
		}
		
		myHashMap aHashMap = new myHashMap();
		
		int serverPort = 0;
		try {
			serverPort = Integer.parseInt(args[0]);
		} catch (Exception e) {
			myLog.error("TCPServer started failed: arguments error.");
			System.out.println("Usage: TCPServerMain <port number>");
			System.exit(0);
		}
		
		ServerSocket aSocket = null;
		try {
			aSocket = new ServerSocket(serverPort);
			myLog.normal("TCPServer opened on " + serverPort);
		} catch (IOException e) {
			myLog.error("TCPServer openning failed on " + serverPort);
			System.out.println("IO: " + e.getMessage());
		} 

		try {
			while (true) {
				
				// Receive client's request
				Socket acceptedSocket = aSocket.accept();
				
				DataInputStream serverIn = new DataInputStream(acceptedSocket.getInputStream());
				DataOutputStream serverOut = new DataOutputStream(acceptedSocket.getOutputStream());

				String serverInput = serverIn.readUTF();

				// Send server's reply
				String serverResponse = aHashMap.requestHandler(serverInput, acceptedSocket.getInetAddress().getHostAddress(), acceptedSocket.getPort());
				serverOut.writeUTF(serverResponse);			
			} 
		} catch (SocketTimeoutException e) {
				myLog.error("Socket Timeout Exception");
				System.out.println("Socket Timeout Exception: " + e.getMessage());
			} catch (SocketException e) {
				myLog.error("Socket Exception");
				System.out.println("Socket: " + e.getMessage());
			} catch (IOException e) {
				myLog.error("IO Exception");
				System.out.println("IO: " + e.getMessage());
				System.exit(0);
			} finally {
				if (aSocket != null)
					try {
						aSocket.close();
					} catch (IOException e) {
						// TODO Auto-generated catch block
						e.printStackTrace();
					}
		}
				
	}
}
