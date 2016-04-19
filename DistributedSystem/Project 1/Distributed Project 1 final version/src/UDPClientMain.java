

import java.net.*;
import java.io.*;

public class UDPClientMain {

	/*
	 * args[0] gives operation contents
	 * args[1] gives server's IP address or hostname
	 * args[2] gives server's port
	 */
	public static void main(String args[]) {
		
		try {
			myLog.createLog("UDPClientLog.txt");
		} catch (IOException e) {
			System.out.println("Error: Log creation for UDP client failed.");
		}
		
		String clientOperation = null;
		InetAddress serverHost = null;
		int serverPort = 0;
		try {	
			clientOperation = args[0];
			// Accept either IP address or hostname
			serverHost = InetAddress.getByName(args[1]);
			serverPort = Integer.parseInt(args[2]);
		} catch (Exception e) {
			myLog.error("UDPClient started failed: arguments error.");
			System.out.println("Usage: UDPClientMain <operation> <server IP address or hostname> <server port>");
			System.exit(0);
		}
		
		
		// Judge: Is this operation legal?
		String[] operationSplitted = clientOperation.split("/");	
		// Allow case-insensitive input
		String operationChecking = operationSplitted[0].toLowerCase().trim();
		if (!operationChecking.equals("put") && !operationChecking.equals("get") && !operationChecking.equals("delete")) {
			myLog.error("UDPClient started failed: invalid operation.");
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


		DatagramSocket clientSocket = null;

		try {
			clientSocket = new DatagramSocket();
			// Set the timeout
			clientSocket.setSoTimeout(10000);
			byte[] clientOperationBytes = clientOperation.getBytes();
			DatagramPacket clientRequest = 
					new DatagramPacket(clientOperationBytes, clientOperation.length(), serverHost, serverPort);
			clientSocket.send(clientRequest);
			
			// Wait for reply
			byte[] clientBuffer = new byte[1024];
			DatagramPacket serverReply = new DatagramPacket(clientBuffer, clientBuffer.length);
			clientSocket.receive(serverReply);
			String serverReplyString = new String(serverReply.getData()).trim();
			myLog.normal("[Server Reply]" + serverReplyString);
			System.out.println("Server reply: " + serverReplyString);
	
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
				clientSocket.close();
		}
	}
	
}
