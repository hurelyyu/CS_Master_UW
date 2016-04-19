import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.io.PrintWriter;
import java.net.Socket;
import java.net.UnknownHostException;

public class TCPClient implements KVServiceProtocolInterface{

	Socket clientSocket;
	PrintWriter tx;
	BufferedReader rx;
	public TCPClient(String host, int port)throws IOException{
		clientSocket=new Socket(host, port);
		tx= new PrintWriter(clientSocket.getOutputStream(),true);
		rx= new BufferedReader(new InputStreamReader(clientSocket.getInputStream()));
		clientSocket.setSoTimeout(10000);
	}
	@Override
	public void processRequest() throws IOException {
		// TODO Auto-generated method stub

	}

	@Override
	public void put(String the_key, String the_value) throws IOException {
		// TODO Auto-generated method stub
		String putOut=Request.PUT+","+the_key+","+the_value;
		tx.println(putOut);
		System.out.println(rx.readLine());
	}

	@Override
	public void get(String the_key) throws IOException {
		// TODO Auto-generated method stub
		String getOut=Request.GET+","+the_key;
		tx.println(getOut);
		System.out.println(rx.readLine());
		System.out.println(rx.readLine());
	}

	@Override
	public void delete(String the_key) throws IOException {
		// TODO Auto-generated method stub
		String putOut=Request.DELETE+","+the_key;
		tx.println(putOut);
		System.out.println(rx.readLine());
	}
	public void close()throws IOException{
		clientSocket.close();
	}
	public static void main(String[] args) {
		// TODO Auto-generated method stub
		if(args.length !=2){
			System.err.println("Usage: java TCPClient <hostname> <port number>");
			System.exit(1);
		}
		String hostName=args[0];
		int portNumber=Integer.parseInt(args[1]);
		try{
			TCPClient client=new TCPClient(hostName,portNumber);
			BufferedReader stdIn=new BufferedReader( new InputStreamReader(System.in));	

			client.get("r2");
			client.put("r1","return1");
			client.put("r2","return2");
			client.put("r3","return3");
			client.put("r4","return4");

			client.get("r1");
			client.get("r2");
			client.get("r5");

			client.put("r2",":)");
			client.delete("r1");
			client.get("r1");
			client.close();
		}catch(UnknownHostException e){
			System.err.println(hostName+"can't be used");
			System.exit(1);
		}catch(IOException e){
			System.err.println("Couldn't get I/O for the connection to "+ hostName);
			System.exit(1);
		}
	}

}
