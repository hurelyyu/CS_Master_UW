
import java.io.IOException;
import java.net.DatagramPacket;
import java.net.DatagramSocket;
import java.net.InetAddress;
import java.net.SocketException;
import java.net.SocketTimeoutException;
import java.util.HashMap;
import java.util.Map;


public class UDPServerThread extends Thread 
implements KVServiceProtocolInterface {
	private int TRIES = 3;

	private int TIMEOUT = 5000;

	private DatagramSocket my_socket;

	private DatagramPacket my_packet;

	private byte[] my_data;

	private Map<String, String> my_store;

	private String my_requested_val;

	private Logger my_log;

	private byte[] my_ack = new byte[1];

	public UDPServerThread() throws IOException {
		super("TCPServerThread");

		my_socket = new DatagramSocket(4445);
		my_store = new HashMap<String,String>();	
		my_log = new Logger("UDPServer_log.txt");
		my_log.write("UDPServer running!");
		my_log.close();
	}

	public void run() {
		while(true) {
			byte[] buf = new byte[256];

			//receive request
			my_packet = new DatagramPacket(buf, buf.length);
			my_data = buf;

			//process request
			try {
				processRequest();
			} catch (IOException e) {
				e.printStackTrace();
			}		
		}
	}

	protected void close() {
		my_log.write("Closed!");
		my_log.close();
		my_socket.close();
	}

	@Override
	public void processRequest() throws IOException {
		my_socket.receive(my_packet);
		my_log = new Logger("UDPServer_log.txt");
		my_log.write("request recieved");
		Request r = null;
		String k = null, v = null;

		my_data = my_packet.getData();
		String data = new String(my_data);
		String delimiters = ",";
		String[] tokens = data.split(delimiters);

		if(tokens.length == 3) { 
			r = Request.valueOf(tokens[0]);
			k = tokens[1];
			v = tokens[2];
		} else if(tokens.length == 2) {
			r = Request.valueOf(tokens[0]);
			k = tokens[1];
		}
		switch (r) {
		case PUT:
			put(k,v);
			sendAck(my_ack);
			break;
		case DELETE:
			delete(k);
			sendAck(my_ack);
			break;
		case GET:
			get(k);
			break;
		default:
			//ERROR: Incorrect request format
			System.out.println("ERROR: Bad request from hostname/IP: " 
					+ my_packet.getAddress() + " port: " 
					+ my_packet.getPort());
			break;
		}
		my_log.write(data);
		my_log.close();
	}

	@Override
	public void put(final String the_key, final String the_value) {
		System.out.println(the_key + " " + the_value);
		my_store.put(the_key, the_value);
		System.out.println(my_store.get(the_key));
	}

	@Override
	public void get(final String the_key) throws IOException {
		// TODO Auto-generated method stub
		System.out.println(the_key);
		my_requested_val = my_store.get(the_key);
		System.out.println(my_requested_val);
		my_data = my_requested_val.getBytes();
		//send response
		sendAck(my_data);
		getAck();
	}

	@Override
	public void delete(final String the_key) {
		my_store.remove(the_key);
	}

	private void sendAck(final byte[] the_ack) throws IOException {
		my_log = new Logger("UDPServer_log.txt");
		//byte[] ack = new byte[1];
		InetAddress address = my_packet.getAddress();
		int port = my_packet.getPort();
		my_packet = new DatagramPacket(the_ack, the_ack.length, address, port);
		my_socket.send(my_packet);
		my_log.write("ack sent");
		my_log.close();
	}

	private void getAck() throws IOException {
		my_log = new Logger("UDPServer_log.txt");
		//set timeout
		my_socket.setSoTimeout(TIMEOUT);
		//wait for response
		my_data = new byte[256];
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
		//turn off timeout
		my_socket.setSoTimeout(0);
		my_log.close();
	}

}
