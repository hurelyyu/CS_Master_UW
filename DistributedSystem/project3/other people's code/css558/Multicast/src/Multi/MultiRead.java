package Multi;

import java.io.IOException;
import java.net.DatagramPacket;
import java.net.MulticastSocket;

public class MultiRead implements Runnable {
	private MulticastSocket socket;
	public MultiRead(MulticastSocket socket){
		this.socket=socket;
	}
	@Override
	public void run() {
		// TODO Auto-generated method stub
		try{
			while(true){
				byte[] buff=new byte[1000];
				DatagramPacket rcv=new DatagramPacket(buff,buff.length);
				socket.receive(rcv);
				String message=new String(rcv.getData(),0,rcv.getLength());
				System.out.println(message);

			}
		}catch(IOException e){
			System.out.println(e.getMessage());
		}

	}

}
