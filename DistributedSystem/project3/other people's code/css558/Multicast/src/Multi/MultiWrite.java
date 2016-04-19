package Multi;

import java.io.IOException;
import java.net.DatagramPacket;
import java.net.InetAddress;
import java.net.MulticastSocket;
import java.util.Scanner;

public class MultiWrite implements Runnable{
	private Scanner talk;
	private InetAddress space;
	private MulticastSocket socket;
	public MultiWrite(InetAddress space, MulticastSocket socket) throws IOException{
		talk=new Scanner(System.in);
		this.space=space;
		this.socket=socket;
	}
	@Override
	public void run() {
		// TODO Auto-generated method stub
		try{

			while(true){
				String msg=talk.nextLine();
				DatagramPacket message=new DatagramPacket(msg.getBytes(),msg.length(),space,socket.getLocalPort());
				socket.send(message);
			}
		}catch(IOException e){
			System.out.println(e.getMessage());
		}

	}

}
