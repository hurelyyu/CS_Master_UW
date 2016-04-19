/*
 *Stephen Mosby
 *Kellen Han-Nin Cheng
 *Aqeel S Bin Rustum
 *Nai-Wei Chen
 *CSS558 Sp14 Project2
 */

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
public class KVStore implements KVService , RMItwophasecommit {
	/** The logger used to create and append to the server's log file. */
	private Logger logger;

	/**The hash map used to store and retrieve the key-value parings.*/
	private HashMap<String,String> KVStore;

	/**The hash map used to store and retrieve the key-value parings.*/
	private List<RMItwophasecommit> my_replicated_servers;

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

	public List<RMItwophasecommit> getMy_replicated_servers() {
		return my_replicated_servers;
	}

	public void setMy_replicated_servers(
			List<RMItwophasecommit> my_replicated_servers) {
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
		if(tpc(the_key, the_value)){
			KVStore.put(the_key, the_value);
			logger.log("Server call: put <" + the_key + "," 
					+ the_value + ">" , true);
		}
		else{
			logger.log("Server call: failed to do TPC -> put <" + the_key + "," 
					+ the_value + ">" , true);
		}
	}

	/**
	 * {@inheritDoc}
	 */
	@Override
	synchronized public void delete(String the_key) throws RemoteException{
		if(tpc(the_key)){
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
		}
		else{
			logger.log("failed deletion of key = " + the_key + " due to failed TPC", true);
		}
	}

	@Override
	public String tpcIsFresh(String the_version) throws RemoteException {
		// TODO Auto-generated method stub
		return null;
	}

	@Override
	public String tpcUpdate(KVStore the_KV_store) throws RemoteException {
		// TODO Auto-generated method stub
		return null;
	}

	@Override
	public int tpcPut(String the_request_id, String... the_request)
			throws RemoteException {
		// TODO Auto-generated method stub
		return 0;
	}

	@Override
	public int tpcDelete(String the_request_id, String... the_request)
			throws RemoteException {
		// TODO Auto-generated method stub
		return 0;
	}

	@Override
	synchronized public boolean tpcACK(String the_request_id, String... the_request)
			throws RemoteException {
		// TODO Auto-generated method stub
		requests.put(the_request_id, the_request);
		return true;
	}

	@Override
	public boolean tpcGO(String the_request_id) throws RemoteException {
		// TODO Auto-generated method stub
		String[] the_request = requests.get(the_request_id);
		if(the_request.length > 1){
			// then it is a put request
			put(the_request[0], the_request[1]);
			requests.remove(the_request_id);
			return true;
		}
		else if(requests.get(the_request_id).length == 1){
			// then it is a delete request
			delete(the_request[0]);
			requests.remove(the_request_id);
			return true;
		}
		else
			return false;
	}

	private String generateId() throws UnknownHostException{
		return Inet4Address.getLocalHost() + "-" + Logger.getTimestamp();
	}

	public boolean tpc(final String...args){
		// TODO Auto-generated method stub
		try{
			boolean succeeded = false;
			boolean [] acks = new boolean[4];
			final String the_request_id = generateId();
			secheduleTask(RequestType.ACK, the_request_id, -1 , acks , args);

			int trials = 0;
			boolean missingACK = false;
			while(!missingACK && trials < 5){
				missingACK = false;
				for(int j = 0 ; j < 5 ; j++){
					if(acks[j] = false){
						missingACK = true;
						secheduleTask(RequestType.ACK,the_request_id, j , acks , args);
					}
				}
				trials++;
			}

			if(!missingACK){
				//do the second phase
				acks = new boolean[4];
				secheduleTask(RequestType.GO,the_request_id, -1 , acks , args);

				trials = 0;
				missingACK = false;
				succeeded = false;
				while(!succeeded && trials < 5){
					missingACK = false;
					for(int j = 0 ; j < 5 ; j++){
						if(acks[j] = false){
							missingACK = true;
							secheduleTask(RequestType.GO,the_request_id, j , acks , args);
						}
					}
					if(! missingACK){
						succeeded = true;
					}
					else
					{
						trials++;
					}
				}
				return succeeded;
			}
			else{
				return false;
			}
		}catch(Exception e){
			e.printStackTrace();
			return false;
		}
	}

	private void secheduleTask(RequestType request, final String the_request_id, 
			final int index, boolean [] acks , final String...args)throws InterruptedException, ExecutionException{
		if(request.equals(RequestType.ACK) && index < 0){
			int i = 0 ;
			for(final RMItwophasecommit rtpc : my_replicated_servers){
				@SuppressWarnings("unchecked")
				RunnableFuture f = new FutureTask(new Callable<Boolean>(){
					// implement call
					public Boolean call() throws RemoteException {
						if(args.length == 2){
							return rtpc.tpcACK(the_request_id, args);
						}
						else{
							return rtpc.tpcACK(the_request_id, args);
						}
					}
				});
				// start the thread to execute it (you may also use an Executor)
				new Thread(f).start();
				// get the result
				try{
					acks[i] = ((Boolean) f.get(1, TimeUnit.SECONDS)).booleanValue();
				}catch (Exception e) {
					logger.log("Timeout", true);
					f.cancel(true);
				}
				i++;
			}
		}
		else if (request.equals(RequestType.ACK) && index >= 0){ 
			// single task
			@SuppressWarnings("unchecked")
			RunnableFuture f = new FutureTask(new Callable<Boolean>(){
				// implement call
				public Boolean call() throws RemoteException {
					if(args.length == 2){
						return my_replicated_servers.get(index).tpcACK(the_request_id, args);
					}
					else{
						return my_replicated_servers.get(index).tpcACK(the_request_id, args);
					}
				}
			});
			// start the thread to execute it (you may also use an Executor)
			new Thread(f).start();
			// get the result
			try{
				acks[index] = ((Boolean) f.get(1, TimeUnit.SECONDS)).booleanValue();
			}catch (Exception e) {
				logger.log("Timeout", true);
				f.cancel(true);
			}

		}
		else if(request.equals(RequestType.GO) && index < 0){
			int i = 0;
			for(final RMItwophasecommit rtpc : my_replicated_servers){
				@SuppressWarnings("unchecked")
				RunnableFuture f = new FutureTask(new Callable<Boolean>(){
					// implement call
					public Boolean call() throws RemoteException {
						if(args.length == 2){
							return rtpc.tpcGO(the_request_id);
						}
						else{
							return rtpc.tpcGO(the_request_id);
						}
					}
				});
				// start the thread to execute it (you may also use an Executor)
				new Thread(f).start();
				// get the result
				try{
					acks[i] = ((Boolean) f.get(1, TimeUnit.SECONDS)).booleanValue();
				}catch (Exception e) {
					logger.log("Timeout", true);
					f.cancel(true);
				}
				i++;
			}
		}
		else if(request.equals(RequestType.GO) && index >= 0){
			// single task
			@SuppressWarnings("unchecked")
			RunnableFuture f = new FutureTask(new Callable<Boolean>(){
				// implement call
				public Boolean call() throws RemoteException {
					if(args.length == 2){
						return my_replicated_servers.get(index).tpcGO(the_request_id);
					}
					else{
						return my_replicated_servers.get(index).tpcGO(the_request_id);
					}
				}
			});
			// start the thread to execute it (you may also use an Executor)
			new Thread(f).start();
			// get the result
			try{
				acks[index] = ((Boolean) f.get(1, TimeUnit.SECONDS)).booleanValue();
			}catch (Exception e) {
				logger.log("Timeout", true);
				f.cancel(true);
			}
		}
	}
}