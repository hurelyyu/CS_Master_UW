/*
 *Stephen Mosby
 *Kellen Han-Nin Cheng
 *Aqeel S Bin Rustum
 *Nai-Wei Chen
 *CSS558 Sp14 Project3
 */

import java.io.IOException;
import java.net.Inet4Address;
import java.rmi.RemoteException;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Random;
import java.util.concurrent.BlockingQueue;
import java.util.concurrent.ConcurrentMap;

/**
 * This is a simple class that implements a key-value store.  Instances of this
 * class reside on the server.
 */
public class KVStore implements KVService, IPaxos, IKVProcessor {
	private static int QUORUM_COUNT = 3;

	private static int SERVER_COUNT = 5;

	private static int RETRIES = 5;

	/** The logger used to create and append to the server's log file. */
	private Logger logger;

	/**The hash map used to store and retrieve the key-value pairings.*/
	private HashMap<String,String> my_KVStore;

	/**The hash map used to store and retrieve the key-value pairings.*/
	private List<IPaxos> my_replicated_servers;

	/**The hash map used to store and retrieve the key-value pairings.*/
	private Map<String, String[]> requests;

	private ConcurrentMap<Float,String> my_log;

	private int my_firstUnchosenIndex;

	private BlockingQueue<String> requests_queue;

	private int server_id;

	private Acceptor acceptor;

	private Random my_rand;


	/**
	 * A simple constructor
	 * @throws IOExceptions
	 */
	public KVStore(int server_id,
			BlockingQueue<String> requests_queue,
			ConcurrentMap<Float, String> the_log) throws IOException{
		this.server_id = server_id;
		my_KVStore=new HashMap<String,String>();
		logger = new Logger("server.log");
		logger.log("RPC.Server start running on : " 
				+ Inet4Address.getLocalHost() , true);
		requests = new HashMap<String, String[]>();
		this.requests_queue = requests_queue;
		my_log = the_log;
		acceptor = new Acceptor(my_log);
	}

	public List<IPaxos> getMy_replicated_servers() {
		return my_replicated_servers;
	}

	public void setMy_replicated_servers(
			final List<IPaxos> my_replicated_servers) {
		my_replicated_servers.add(this);
		this.my_replicated_servers = my_replicated_servers;
	}

	public Map<String, String[]> getRequests() {
		return requests;
	}

	public void addRequest(String the_request){
		requests_queue.add(the_request);

	}

	/**
	 * {@inheritDoc}
	 */
	@Override
	synchronized public String get(String the_key) throws RemoteException{
		logger.log("Server call: get (" + the_key + ")", true);
		//		System.out.println("Server call: get (" + the_key + ")");
		String value = my_KVStore.get(the_key);
		logger.log("key = " + the_key + " , value = " + value, true);
		//		System.out.println("key = " + the_key + " , value = " + value);
		return value;
	}

	/**
	 * {@inheritDoc}
	 */
	@Override
	synchronized public void put(String key, String value) 
			throws RemoteException{
		String the_request = "put" + " " + key + " " + value;
		addRequest(the_request);
	}

	/**
	 * {@inheritDoc}
	 */
	@Override
	synchronized public void delete(String the_key) throws RemoteException{
		String the_request = "delete" + " " + the_key ;
		addRequest(the_request);
	}



	//	/*
	//	 * If an acceptor receives a prepare request with number n greater
	//	 * than that of any prepare request to which it has already responded,
	//	 * the it responds to the request with a promise not to accept any more
	//	 * proposals numbered less than n and with the highest-numbered proposal
	//	 * (if any) that it has accepted.
	//	 */
	//
	//	/**
	//	 * {@inheritDoc}
	//	 */
	//	@Override
	//	public String checkForLeader(int server_id) {
	//		// TODO Auto-generated method stub
	//		return null;
	//	}

	/**
	 * {@inheritDoc}
	 * @throws InterruptedException 
	 */
	@Override
	public String prepare(float n) throws RemoteException, InterruptedException {
		int id = (int)(n * 10);
		id = id % 10;
		System.out.println("id = " + id);
		System.out.println("server_id = " + server_id);
		if(server_id != id){
			int rand = my_rand.nextInt(100);
			System.out.print("sleep for " + rand + " seconds");
			if(rand < 10){
				Thread.sleep((5)*1000);
			}
		}
		return acceptor.prepare(n); 
	}

	/*
	 * If an acceptor receives an accept request for a proposal numbered n, it
	 * accepts the proposal unless it has already responded to a prepare request
	 * have a number greater than n.
	 */
	@Override
	public String accept(float n, String value)
			throws RemoteException {
		String s=acceptor.accept(n, value);
		System.out.println(s + " = my_log: " + n + " " + my_log.get(n));
		return s;
	}



	@Override
	public void learn(String accepted_proposal_and_value)
			throws RemoteException{
		String[] tokens = accepted_proposal_and_value.split(",");
		float proposal_number = Float.parseFloat(tokens[0]);
		String value = tokens[1]+","+tokens[2];
		//overwrite the [accepted] log for that request
		//		System.out.println("Before learning: "+my_log.get(proposal_number));
		my_log.put(proposal_number, value);
		//		System.out.println("After learning: "+my_log.get(proposal_number));
		reset_acceptor();
	}

	@Override
	public int getServer_id(){
		return server_id;
	}

	@Override
	public void kv_put(String key, String val) {
		my_KVStore.put(key, val);
		System.out.println(my_KVStore.toString());
	}

	@Override
	public void kv_delete(String key) {
		my_KVStore.remove(key);
		System.out.println(my_KVStore.toString());
	}

	@Override
	public int kv_size() {
		return my_KVStore.size();
	}

	// this will be called after a value has been chosen 
	// so that value and its proposal need to be removed from the acceptor
	public void reset_acceptor(){
		acceptor.setAccepted_proposal_number(0);
		acceptor.setAccepted_value("NO_ACCEPTED_VALUE_YET");
	}
}