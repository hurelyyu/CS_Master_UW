/*
 *Stephen Mosby
 *Kellen Han-Nin Cheng
 *Aqeel S Bin Rustum
 *Nai-Wei Chen
 *CSS558 Sp14 Project2
*/

import java.io.IOException;
import java.net.Inet4Address;
import java.rmi.RemoteException;
import java.util.HashMap;

/**
 * This is a simple class that implements a key-value store.  Instances of this
 * class reside on the server.
 */
public class KVStore implements KVService {
	/** The logger used to create and append to the server's log file. */
	private Logger logger;
	
	/**The hash map used to store and retrieve the key-value parings.*/
	private HashMap<String,String> KVStore;

	/**
	 * A simple constructor
	 * @throws IOException
	 */
	public KVStore() throws IOException{
		KVStore=new HashMap<String,String>();
		logger = new Logger("server.log");
		logger.log("RPC.Server start running on : " 
				+ Inet4Address.getLocalHost() , true);
	}

	/**
	 * {@inheritDoc}
	 */
	@Override
	synchronized public String get(String the_key) throws RemoteException{
		logger.log("Server call: get (" + the_key + ")", true);
		System.out.println("Server call: get (" + the_key + ")");
		String value = KVStore.get(the_key);
		logger.log("key = " + the_key + " , value = " + value, true);
		System.out.println("key = " + the_key + " , value = " + value);
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
		System.out.println("Server call: put <" + the_key + "," 
				+ the_value + ">" );
	}

	/**
	 * {@inheritDoc}
	 */
	@Override
	synchronized public void delete(String the_key) throws RemoteException{
		if(!KVStore.containsKey(the_key)){
			logger.log("Key = " + the_key + " not found");
			System.out.println("Key = " + the_key + " not found");
		} else {
			KVStore.remove(the_key);
			logger.log("Server call: delete(" + the_key + ")", true);
			System.out.println("Server call: delete(" + the_key + ")");
		}
		if(!KVStore.containsKey(the_key)){
			logger.log("successful deletion of key = " + the_key, true);
			logger.log(KVStore.toString(), true);
		} else{
			logger.log("failed deletion of key = " + the_key, true);
		}
	}

}
