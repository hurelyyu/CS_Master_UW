/*
 *Stephen Mosby
 *Kellen Han-Nin Cheng
 *Aqeel S Bin Rustum
 *Nai-Wei Chen
 *CSS558 Sp14 Project3
 */

import java.io.IOException;
import java.net.Inet4Address;
import java.rmi.Naming;
import java.rmi.NotBoundException;
import java.rmi.RemoteException;


/**
 * Implements the client-side code, which calls procedures on remote objects
 * located on the server.
 */
public class RMIClient  implements Runnable {
	/**Specifies the amount of time to sleep a thread in milliseconds */
	private int SLEEP_TIME = 1000*3;
	
	/**The interface for the remote object */
	private KVService kvs;
	
	/**The client thread id */
	private int my_id;
	
	/**The name of the KV server */
	private String my_host;
	
	/**The the logger used by this client thread */
	private Logger logger;

	/**
	 * A simple constructor that locates the registry on the specified server,
	 * and gets a handle on the remote object for subsequent remote procedure
	 * calls.
	 * @param the_host - the name of the server providing the KV store service
	 * @param the_id - the id of this client thread
	 * @throws NotBoundException
	 * @throws IOException
	 */
	public RMIClient(String the_host, int the_id) 
			throws NotBoundException, IOException {
		String name = "KVService";
		my_id = the_id;
		my_host = java.net.InetAddress.getLocalHost().getHostName();
		//		Registry registry = LocateRegistry.getRegistry(the_host);
		//		kvs=(KVService) registry.lookup(name);
		try{
			kvs = (KVService) Naming.lookup("//"+the_host+"/"+name);
		}catch(Exception e){
			System.out.println(e.getMessage());
		}
		String timestamp = Logger.getTimestamp();
		logger = new Logger("Client-" + the_id +"-" + timestamp + ".log");
		logger.log("Client " + the_id + " is running on : " 
				+ Inet4Address.getLocalHost(), true);
		System.out.println("Client " + the_id 
				+ " is running on : " + Inet4Address.getLocalHost());
	}

	/**
	 * A wrapper method for calling the put procedure on the remote object.
	 * @param the_key - the key used in a KVStore
	 * @param the_value - the value used in a KVStore
	 * @throws RemoteException
	 * @throws InterruptedException
	 */
	private void put(final String the_key, final String the_value) 
			throws RemoteException, InterruptedException {
		kvs.put(the_key,the_value);
		System.out.println(my_host + ": Thread " 
				+ my_id + ": put " + the_key + " " + the_value);
		logger.log("Client " + my_id 
				+ " has issued request: put("+the_key+","+the_value+")" , true);
		Thread.sleep(SLEEP_TIME);
	}

	/**
	 * A wrappter method for calling the get procedure on the remote object.
	 * @param the_key - the key used to get a value in a KVStore
	 * @throws RemoteException
	 * @throws InterruptedException
	 */
	private void get(final String the_key) 
			throws RemoteException, InterruptedException {
		String v;
		logger.log("Client " + my_id 
				+ " has issued request: get("+the_key+")" , true);
		v = kvs.get(the_key);
		System.out.println(my_host + ": Thread " + my_id
				+ ": get " + the_key + "; value = " + v);
		logger.log("Client " + my_id + " has got--> key = "
				+ the_key+ " , value = " + v, true);
		Thread.sleep(SLEEP_TIME);
	}

	/**
	 * A wrapper method for calling the delete procedure on the remote object.
	 * @param the_key - the key used to delete a key-value pair in a KVStore
	 * @throws InterruptedException
	 * @throws RemoteException
	 */
	private void delete(final String the_key)
			throws InterruptedException, RemoteException {
		kvs.delete(the_key);
		System.out.println(my_host + ": Thread " + my_id
				+ ": " + the_key + " deleted");
		logger.log("Client " + my_id 
				+ " has issued request: delete("+the_key+")" , true);
		Thread.sleep(SLEEP_TIME);
	}


	@Override 
	public void run() {
		// TODO Auto-generated method stub
		while(true) {
			try{
				put("key1", "value1");
				put("key2", "value2");
				put("key3", "value3");
				put("key4", "value4");
				put("key5", "value5");
				
				get("key1");
				get("key2");
				get("key3");
				get("key4");
				get("key5");

				delete("key1");
				delete("key2");
				delete("key3");
				delete("key4");
				delete("key5");

				get("key1");

			}catch(Exception e){

			}
		}
	}
}


