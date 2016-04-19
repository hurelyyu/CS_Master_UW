// UDPClientHM.java A simple UDP client program.
import java.net.*;
import java.io.*;
import java.util.Date;
import java.text.SimpleDateFormat;

public class UDPClientHM {
	protected DatagramSocket aSocket;
	public void setDatagramSocket(DatagramSocket aSocket) {
		this.aSocket = aSocket;
	}
	// method to send and receive message
	public void sendReceive(DatagramPacket request) {
		try {
			aSocket.send(request);
			byte[] buffer = new byte[1000];
			DatagramPacket reply = new DatagramPacket(buffer,buffer.length);
			aSocket.receive(reply);
			updateClientLog(new String(reply.getData()));
			//System.out.println("Reply: "+new String(reply.getData()));
		}
		catch (IOException e) {
			System.out.println("IO: " + e.getMessage());
			updateClientLog("IO: " + e.getMessage());
		}
		finally {
			if(aSocket != null) 
				aSocket.close();
		}

	}
	// method to generate/update client log for every output
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
    			out.write("\nReply: " + output + "\t" + datetime);
    			System.out.println("Reply: "+output + "\t" + datetime);
    			out.close();
    			System.out.println("Client log is updated as clientlog.txt");
			}
			catch (IOException ioe)
			{
    			System.err.println("Error: The log is not able to be generated");
			}
		}
		else {
			try {
				PrintWriter writelog = new PrintWriter("clientlog.txt", "UTF-8");
				writelog.println("Reply: "+output + "\t" + datetime);
				System.out.println("Reply: "+output + "\t" + datetime);
				writelog.close();
				System.out.println("Client log is generated as clientlog.txt");
			}
			catch (IOException ioe){
				System.err.println("Error: The log is not able to be generated");
			}
		}
	}	
		public static void main(String[] args) {
			if(args.length < 3) {
				System.out.println("Usage: java UDPClientHM <Hostname> <Port number> <Message>");
				System.exit(1);
			}
			try {
				UDPClientHM udpClientHM = new UDPClientHM();
				DatagramSocket aSocket = new DatagramSocket();
				byte[] m = args[2].getBytes();
				InetAddress aHost = InetAddress.getByName(args[0]);
				int serverPort = Integer.valueOf(args[1]).intValue();
				aSocket.setSoTimeout(10000);
				udpClientHM.setDatagramSocket(aSocket);
				System.out.println(args[2]);
				DatagramPacket request = 
					new DatagramPacket(m,m.length,aHost,serverPort);
				udpClientHM.sendReceive(request);
			}
			catch(NumberFormatException e) {
				System.out.println("Invalid port number");
				UpdateClientLog updateClog = new UpdateClientLog("Invalid port number");
			}
			catch (UnknownHostException e) {
				System.out.println("Unknown Host");
				UpdateClientLog updateClog = new UpdateClientLog("Unknown Host");
			}
			catch (SocketException e) {
				System.out.println("Failed to make connection to Server");
				UpdateClientLog updateClog = new UpdateClientLog("Failed to make connection to Server at port: "+Integer.valueOf(args[1]).intValue());
			}
		}
}