import java.io.IOException;
import java.net.InetSocketAddress;
import java.net.Socket;
import java.net.SocketException;

public class TCPclientmain {
	/*
     * args[0] gives request contents
     * args[1] gives server's port number
     * args[2] gives message that send to server
     * 
     */
public static void main(String[] args) {
	
	if(args.length < 3) {
		System.out.println("Usage: java TCPclient <Hostname> <Port Number> <Function key:value>/<Function key>");
	}
	else {
		String hostname = args[0];
		int port = Integer.parseInt(args[1]);
		try {
			
			// create a socket
			// perform a simple operation PUT(1,3)
			Socket socket = new Socket();
			try {
				socket.connect(new InetSocketAddress(hostname,port),10000);
			} catch (IOException e) {
				// TODO Auto-generated catch block
				e.printStackTrace();
			}
			socket.setSoTimeout(10000); //time out set up to 10000
			String sendmessage = args[2];
			TCPclient tcpclient = new TCPclient();
			//
			tcpclient.startSocket(socket);
			tcpclient.sentreceive(sendmessage);
		} catch (SocketException e){
			new clientlog("Failed to connect to the Server at port: ", port);
		}catch (Exception e){
			System.out.println("Fail to connect to the Server at port: "+args[1]);
			new clientlog("Failed to connect to the Server at port: ", port);
			
		}

		
		
	}
	}

	
}
