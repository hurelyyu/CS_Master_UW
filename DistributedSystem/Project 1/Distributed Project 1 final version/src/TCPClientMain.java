

import java.io.*;
import java.net.*;

public class TCPClientMain {
	
	/*
	 * args[0] gives operation contents
	 * args[1] gives server's IP address or hostname
	 * args[2] gives server's port
	 */
	public static void main(String args[]) {
		
		try {
			myLog.createLog("TCPClientLog.txt");
		} catch (IOException e) {
			System.out.println("Error: Log creation for TCP client failed.");
		}
		
		String clientOutput = null;
		InetAddress serverHost = null;
		int serverPort = 0;
		try {	
			clientOutput = args[0];
			// Accept either IP address or hostname
			serverHost = InetAddress.getByName(args[1]);
			serverPort = Integer.parseInt(args[2]);
		} catch (Exception e) {
			myLog.error("TCPClient started failed: arguments error.");
			System.out.println("Usage: TCPClientMain <operation> <server IP address or hostname> <server port>");
			System.exit(0);
		}
		
		// Judge: Is this operation legal?
		String[] operationSplitted = clientOutput.split("/");	
		// Allow case-insensitive input
		String operationChecking = operationSplitted[0].toLowerCase().trim();
		if (!operationChecking.equals("put") && !operationChecking.equals("get") && !operationChecking.equals("delete")) {
			myLog.error("TCPClient started failed: invalid operation.");
			System.out.println("Invalid operation: "+operationChecking + "\r\nAvaliable operation: put/<key>/<value>, get/<key>, delete/<key>.");
			System.exit(0);
		}
		
		if (operationChecking.equals("put")) {
			myLog.normal(" Operation: "+ operationSplitted[0].toLowerCase().trim() + ": (" + 
					operationSplitted[1].toLowerCase().trim() + "," +
					operationSplitted[2].toLowerCase().trim() + ")");
		} else {
			myLog.normal(" Operation: "+ operationSplitted[0].toLowerCase().trim() + ": " + 
					operationSplitted[1].toLowerCase().trim());
		}


		Socket clientSocket = null;

		try {
			clientSocket = new Socket(serverHost, serverPort);
			// Set the timeout
			clientSocket.setSoTimeout(10000);
			
			DataInputStream clientIn = new DataInputStream(clientSocket.getInputStream());
			DataOutputStream clientOut = new DataOutputStream(clientSocket.getOutputStream());

			clientOut.writeUTF(clientOutput);
			
			// Receive the reply from server
			String clientInput = new String();
			clientInput = clientIn.readUTF();
			
			myLog.normal("[Server Reply]" + clientInput);
			System.out.println("Server reply: " + clientInput);
	
		} catch (SocketTimeoutException e) {
			// Note the timeout exception in log
			myLog.error("Socket Timeout Exception");
			System.out.println("Socket Timeout Exception: " + e.getMessage());
		} catch (SocketException e) {
			myLog.error("Socket Exception");
			System.out.println("Socket: " + e.getMessage());
		} catch (IOException e) {
			myLog.error("IO Exception");
			System.out.println("IO: " + e.getMessage());
		} finally {
			if (clientSocket != null)
				try {
					clientSocket.close();
				} catch (IOException e) {
					// TODO Auto-generated catch block
					e.printStackTrace();
				}
		}
	}

	

}
