/**
 * Created by Aqeel Bin Rustum on 4/22/14.
 */

import java.io.IOException;
import java.net.Inet4Address;
import java.rmi.Naming;
import java.rmi.RemoteException;
import java.util.Scanner;

public class Client
{
	private Logger logger ;
	private Server server;
	private String clientID;
	public Client(Server server, String clientID) throws IOException {
		this.server = server;
		this.clientID = clientID;
		String timestamp = Logger.getTimestamp();
		logger = new Logger("Client-" + clientID +"-" + timestamp + ".log");
		logger.log("Client " + clientID + " is running on : " + Inet4Address.getLocalHost(), true);
	}

	public void put(String key, String value){
		logger.log("Client " + clientID + " has issued request: put("+key+","+value+")" , true);
		try {
			this.server.put(key, value);
		} catch (RemoteException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
	}

	public void get(String key){
		logger.log("Client " + clientID + " has issued request: get("+key+")" , true);
		String value = null;
		try {
			value = this.server.get(key);
			logger.log("Client " + clientID + " has got--> key = " + key+ " , value = " + value, true);
		} catch (RemoteException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
	}

	public void delete(String key){
		logger.log("Client " + clientID + " has issued request: delete("+key+")" , true);
		try {
			this.server.delete(key);
		} catch (RemoteException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}

	}

	public void prompt() throws RemoteException {
		Scanner input = new Scanner(System.in);
		String [] request = null;
		while(true){
			System.out.print("Enter your request: ");
			request = input.nextLine().trim().split("\\s+");
			if(request.length >= 2){
				if(request[0].toLowerCase().equals("put")){
					if(request.length == 3){
						put(request[1],request[2]);
					}
					else{
						System.out.println("Invalid request");
					}
				}
				else if(request[0].toLowerCase().equals("get")){
					get(request[1]);
				}
				else if(request[0].toLowerCase().equals("delete")){
					delete(request[1]);
				}
			}
			else if(request.length == 1 && request[0].toLowerCase().equals("exit")){
				System.exit(0);
				input.close();
			}
			else{
				System.out.println("Invalid request");
			}
		}
	}
	public static void main(String []args){
		String serverName , serverIPAddress ;
		int serverPort = 0 ;
		if(args.length == 2){
			serverIPAddress = args[0];
      serverPort = Integer.parseInt(args[1]);
		}
		else{
      System.out.println("No sufficient parameters");
      System.exit(2);
			//serverPort = 2212;	
		}
    
    serverName = "RMISERVER";
		final String url = "rmi://localhost:" + serverPort + "/" + serverName;
		try {
			
			// client 1
			new Thread(
					new Runnable() {
						public void run() {
							Server server = null;
							try {
								server = (Server) Naming.lookup(url);
								Client client = new Client(server , "1");
								client.delete("pink");
								client.put("black", "2525");
								client.delete("yellow");
								client.put("red", "7548");
								client.put("green", "5555");
								client.get("black");
							} catch (Exception e) {
								// TODO Auto-generated catch block
								e.printStackTrace();
							}
						}
					}).start();
			
			//client 2
			new Thread(
					new Runnable() {
						public void run() {
							Server server = null;
							try {
								server = (Server) Naming.lookup(url);
								Client client = new Client(server , "2");
								client.put("yellow", "8888");
								client.get("red");
								client.delete("black");
								client.get("yellow");
								
							} catch (Exception e) {
								// TODO Auto-generated catch block
								e.printStackTrace();
							}
						}
					}).start();
			
			//client 3
			new Thread(
					new Runnable() {
						public void run() {
							Server server = null;
							try {
								server = (Server) Naming.lookup(url);
								Client client = new Client(server, "3");
								client.put("pink", "1144");
								client.delete("green");
								client.put("orang", "2233");
								client.get("pink");
								client.put("brown", "8965");
							} catch (Exception e) {
								// TODO Auto-generated catch block
								e.printStackTrace();
							}
						}
					}).start();

		} catch (Exception e) {
			e.printStackTrace();
		}
	}
}

