import java.io.IOException;
import java.net.DatagramPacket;
import java.net.DatagramSocket;
import java.net.InetAddress;
import java.net.SocketException;
import java.net.SocketTimeoutException;
import java.net.UnknownHostException;

/**
 * 
 * @author Stephen Mosby
 * @version 4/15/2014
 *
 */
public class UDPClient implements KVServiceProtocolInterface {

	private int TRIES = 3;
	
	private int TIMEOUT = 5000;
	
	private DatagramSocket my_socket;

	private byte[] my_data;

	private InetAddress my_address;

	private int my_port;

	private Logger my_log;

	private byte[] my_ack = new byte[1];

	public UDPClient(final String the_host, final String the_port) 
			throws IOException {
		my_socket = new DatagramSocket();
		my_data = new byte[256];
		my_address = InetAddress.getByName(the_host);
		my_port = Integer.parseInt(the_port);
		my_log = new Logger("client_log.txt");
		my_log.write("UDPClient running!");
		//set timeout
		my_socket.setSoTimeout(TIMEOUT);
	}

	@Override
	public void processRequest() throws IOException {
		//write code here to process console input
	}


	@Override
	public void put(final String the_key, final String the_value) 
			throws IOException {
		String r = Request.PUT + "," + the_key + "," + the_value;
		sendRequest(r);
		getAck();
	}


	@Override
	public void get(final String the_key) throws IOException {
		//send request
		String r = Request.GET + "," + the_key;
		sendRequest(r);

		//wait for response
		DatagramPacket packet = new DatagramPacket(my_data, my_data.length);
		int tries = 3;
		while (tries>0 ) {
			try {
				my_socket.receive(packet);
				my_log.write("Recieved value: " + new String(packet.getData()));
				//send ack
				packet = new DatagramPacket(my_ack, my_ack.length, my_address, 
						my_port);
				my_socket.send(packet);
				break;
			} catch(SocketTimeoutException e) {
				tries--;
				if(tries == 0){
					my_log.write("Server timed out!  ...moving on");
				}  else {
					my_log.write("Server timed out! ...trying again");
				}
			}
		}
	}


	@Override
	public void delete(final String the_key) throws IOException {
		String r = Request.DELETE + "," + the_key;
		sendRequest(r);
		getAck();
	}
	
	private void sendRequest(final String the_request) throws IOException {
		my_data = the_request.getBytes();
		DatagramPacket packet = new DatagramPacket(my_data, my_data.length,
				my_address, my_port);
		my_socket.send(packet);
		my_log.write(the_request);
	}
	
	private void close() {
		my_log.write("Closed!");
		my_log.close();
		my_socket.close();
	}
	
	private void getAck() throws IOException {
		//wait for response
		DatagramPacket packet = new DatagramPacket(my_data, my_data.length);
		int tries = TRIES;
		while (tries>0 ) {
			try {
				my_socket.receive(packet);
				my_log.write("ACK recieved");
				break;
			} catch(SocketTimeoutException e) {
				tries--;
				if(tries == 0){
					my_log.write("Server timed out!  ...moving on");
				}  else {
					my_log.write("Server timed out! ...trying again");
				}
			}
		}
	}

	public static void main(String[] args) throws IOException {
		if(args.length != 2) {
			System.out.println("Usage: java UDPClient <hostname> <port>");
		}
		
		UDPClient c = new UDPClient(args[0], args[1]);
		c.put("key1", "value1");
		c.put("key2", "value2");
		c.put("key3", "value3");
		c.put("key4", "value4");
		c.put("key5", "value5");
		c.get("key1");
		c.get("key2");
		c.get("key3");
		c.get("key4");
		c.get("key5");
		c.delete("key1");
		c.delete("key2");
		c.delete("key3");
		c.delete("key4");
		c.delete("key5");
		c.close();
	}
}
