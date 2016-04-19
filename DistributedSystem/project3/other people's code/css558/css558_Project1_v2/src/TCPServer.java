

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.io.PrintWriter;
import java.net.Inet4Address;
import java.net.ServerSocket;
import java.net.Socket;
import java.util.HashMap;

public class TCPServer extends Thread implements KVServiceProtocolInterface{

	static HashMap<String,String> storage=new HashMap<String,String>();
	PrintWriter tx;
	BufferedReader rx;
	Socket client;

	public TCPServer(Socket socket)throws IOException{
		super ("TCPServer");
//		storage=new HashMap<String,String>();
		client=socket;
		System.out.println(Inet4Address.getLocalHost());
	}
	@Override
	public void processRequest() throws IOException {
		// TODO Auto-generated method stub

		while(true){
			String req=rx.readLine();
			if(req!=null){
				processSingleRequest(req);
			}
		}	
				


	}

	private void processSingleRequest(String req)throws IOException{
		tx.println(req);
		System.out.println(req);
		String[] decomp=req.split(",");
		switch(decomp[0]){
		case "PUT":
			put(decomp[1],decomp[2]);
			break;
		case "GET":
			get(decomp[1]);
			break;
		case "DELETE":
			delete(decomp[1]);
			break;
		default:
			System.out.println("unknown command");
		}
	}

	@Override
	public void put(String the_key, String the_value) throws IOException {
		// TODO Auto-generated method stub
		storage.put(the_key, the_value);

	}

	@Override
	public void get(String the_key) throws IOException {
		// TODO Auto-generated method stub
		tx.println(storage.get(the_key));
	}

	@Override
	public void delete(String the_key) throws IOException {
		// TODO Auto-generated method stub
		storage.remove(the_key);
	}

	public void run(){
		while(true){
			try{
				
				tx=new PrintWriter(client.getOutputStream(),true);
				rx =new BufferedReader(new InputStreamReader(client.getInputStream()));
				processRequest();
				tx.close();
				rx.close();
				client.close();

			}catch(IOException e){
				System.out.println("Connection unavailable");

			}
		}

	}


	public static void main(String[] args) throws IOException {
		// TODO Auto-generated method stub
		if(args.length != 1){
			System.err.print("Usage: java TCPServer <portnumber>");
			System.exit(1);
		}
		int portnumber=Integer.parseInt(args[0]);
//		HashMap<String,String> storage=new HashMap<String,String>();
		try(ServerSocket server=new ServerSocket(portnumber)){
			while(true){
				new TCPServer(server.accept()).start();
			}
		}
		catch(IOException e){
			System.out.println("Exception caught when trying to listen on port "+portnumber+" or listening for a connection");
			System.out.println(e.getMessage());
		}


	}
}
