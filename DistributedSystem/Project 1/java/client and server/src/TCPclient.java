
import java.net.*;
import java.io.*;
import java.util.HashMap;
import java.util.Date;
import java.text.*;
import java.util.logging.*;

public class TCPclient {
	// set up a new socket
	Socket socket;
	public void startSocket(Socket socket){
		this.socket = socket;
	}
	//send out message
	public void sentreceive(String MessageSend){
	try{
		BufferedWriter writer = new BufferedWriter(
			new OutputStreamWriter(socket.getOutputStream()));
		writer.write(MessageSend);
		writer.newLine();
		writer.flush();
	//get result from server
		BufferedReader reader = new BufferedReader(
			new InputStreamReader(socket.getInputStream()));
		String readline = reader.readLine();
		System.out.println("Readline is: " + reader.readLine());
		clientlog updateclientlog = new clientlog(readline,(Integer) null);
		reader.close();
		writer.close();	
	}
		catch (Exception e) {
			System.out.println("Error occur during recieving");
		}
	}

}
