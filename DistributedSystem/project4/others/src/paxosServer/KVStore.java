package paxosServer;


import java.io.IOException;
import java.net.Inet4Address;
import java.net.UnknownHostException;
import java.rmi.RemoteException;
import java.util.*;
import java.util.concurrent.Callable;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.FutureTask;
import java.util.concurrent.RunnableFuture;
import java.util.concurrent.TimeUnit;

/**
 * This is a simple class that implements a key-value store.  Instances of this
 * class reside on the server.
 */
public class KVStore implements KVService , Paxos {
	/** The logger used to create and append to the server's log file. */
	private Logger logger;

	/**The hash map used to store and retrieve the key-value parings.*/
	private HashMap<String,String> KVStore;

	/**The hash map used to store and retrieve the key-value parings.*/
	private List<Paxos> my_replicated_servers;

	/**The hash map used to store and retrieve the key-value parings.*/
	private Map<String, String[]> requests;

	public enum RequestType {ACK, GO}
	/**
	 * A simple constructor
	 * @throws IOExceptions
	 */
	public KVStore() throws IOException{
		KVStore=new HashMap<String,String>();
		logger = new Logger("server.log");
		logger.log("RPC.Server start running on : " 
				+ Inet4Address.getLocalHost() , true);
		requests = new HashMap<String, String[]>();
	}

	public List<Paxos> getMy_replicated_servers() {
		return my_replicated_servers;
	}

	public void setMy_replicated_servers(
			List<Paxos> my_replicated_servers) {
		this.my_replicated_servers = my_replicated_servers;
	}

	public Map<String, String[]> getRequests() {
		return requests;
	}

	synchronized public void addRequest(String the_request_id, String...the_request){
		requests.put(the_request_id, the_request);
	}

	synchronized public void deleteRequest(String the_request_id){
		requests.remove(the_request_id);
	}


	/**
	 * {@inheritDoc}
	 */
	@Override
	synchronized public String get(String the_key) throws RemoteException{
		logger.log("Server call: get (" + the_key + ")", true);
		//		System.out.println("Server call: get (" + the_key + ")");
		String value = KVStore.get(the_key);
		logger.log("key = " + the_key + " , value = " + value, true);
		//		System.out.println("key = " + the_key + " , value = " + value);
		return value;
	}

	/**
	 * {@inheritDoc}
	 */
	@Override
	synchronized public void put(String the_key, String the_value) 
			throws RemoteException{

			KVStore.put(the_key, the_value);
			logger.log("Server call: put <" + the_key + "," 
					+ the_value + ">" , true);
		
		logger.log(KVStore.toString(),true);
	}

	/**
	 * {@inheritDoc}
	 */
	@Override
	synchronized public void delete(String the_key) throws RemoteException{

		if(!KVStore.containsKey(the_key)){
			logger.log("Key = " + the_key + " not found");
		} else {
			KVStore.remove(the_key);
			logger.log("Server call: delete(" + the_key + ")", true);
		}
		if(!KVStore.containsKey(the_key)){
			logger.log("successful deletion of key = " + the_key, true);
			logger.log(KVStore.toString(), true);
		} else{
			logger.log("failed deletion of key = " + the_key, true);
		}


		logger.log(KVStore.toString(),true);
	}

	@Override
	public void prepare() throws RemoteException {
		// TODO Auto-generated method stub
		
	}

	@Override
	public void accept() throws RemoteException {
		// TODO Auto-generated method stub
		
	}

	@Override
	public void commit() throws RemoteException {
		// TODO Auto-generated method stub
		
	}
}