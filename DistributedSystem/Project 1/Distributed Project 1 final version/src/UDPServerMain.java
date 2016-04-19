

import java.net.*;
import java.io.*;

public class UDPServerMain {
	/*
	 * args[0] gives server's port
	 */
	public static void main(String args[]) {

		try {
			myLog.createLog("UDPServerLog.txt");
		} catch (IOException e) {
			System.out.println("Error: Log creation for UDP server failed.");
		}
		
		myHashMap aHashMap = new myHashMap();
		
		int serverPort = 0;
		try {
			serverPort = Integer.parseInt(args[0]);
		} catch (Exception e) {
			myLog.error("UDPServer started failed: arguments error.");
			System.out.println("Usage: UDPServerMain <port number>");
			System.exit(0);
		}
		
		DatagramSocket serverSocket = null;
		try {
			serverSocket = new DatagramSocket(serverPort);
			myLog.normal("UDPServer opened on " + serverPort);
		} catch (IOException e) {
			myLog.error("UDPServer openning failed on " + serverPort);
			System.out.println("IO: " + e.getMessage());
		} 

		try {
			while (true) {
				
				// Receive client's request
				byte[] serverBuffer = new byte[1024];
				DatagramPacket clientRequest = new DatagramPacket(serverBuffer, serverBuffer.length);
				serverSocket.receive(clientRequest);
				
				// Execute requested operation
				String requestContent = new String(clientRequest.getData());

				String serverResponse = aHashMap.requestHandler(requestContent, clientRequest.getAddress().getHostAddress(), clientRequest.getPort());
				byte[] serverResponseBytes = serverResponse.getBytes();
				
				// Send server's reply
				DatagramPacket serverReply = new DatagramPacket(
						serverResponseBytes, serverResponseBytes.length, 
						clientRequest.getAddress(), clientRequest.getPort());
				serverSocket.send(serverReply);
				
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
				if (serverSocket != null)
					serverSocket.close();
		}
				
	}
}
