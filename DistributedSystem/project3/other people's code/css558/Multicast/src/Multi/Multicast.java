package Multi;

import java.net.InetAddress;
import java.net.MulticastSocket;

public class Multicast {

	public static void main(String[] args) {
		// TODO Auto-generated method stub
		try{
//			Scanner talk=new Scanner(System.in);
			
			InetAddress group=InetAddress.getByName("228.5.6.7");
			MulticastSocket socket=new MulticastSocket(8008);
			socket.joinGroup(group);
			
			System.out.println("Joined Multicast Channel "+ group.getHostAddress()+" on port "+socket.getLocalPort());
			new Thread(new MultiRead(socket)).start();
			new Thread(new MultiWrite(group,socket)).start();
//			while(true){
//				String msg= talk.nextLine();
//				DatagramPacket message=new DatagramPacket(msg.getBytes(),msg.length(),group,8008);
//				socket.send(message);
//				byte[] buff=new byte[1000];
//				DatagramPacket recv=new DatagramPacket(buff, buff.length);
//				socket.receive(recv);
//				String received=new String(recv.getData(),0,recv.getLength());
//				System.out.println(received);
//			}
			
		}catch(Exception e){
			System.out.println(e.getMessage());
			System.exit(0);
		}
	}

}
