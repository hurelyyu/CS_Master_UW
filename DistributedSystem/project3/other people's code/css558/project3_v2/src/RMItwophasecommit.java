import java.rmi.Remote;
import java.rmi.RemoteException;

/**
 * This is a simple interface, describing the methods to be implemented 
 * server-side for providing consistency among replicated servers using 2-phase
 * commit.  The methods are called by a server who wants to commit its changes
 * to the replicated servers on the remote objects processing these calls 
 * located on the other replicated servers.
 */
public interface RMItwophasecommit extends Remote{
	
	/**
	 * This is a put request to be replicated on the replicated
	 * servers. This is the first message associated with the first phase
	 * in the 2PC.
	 * @param the_request_id - this is the id associated with this request
	 * @param the_request - the put call to be replicated in string form
	 * @return - the ACK message
	 */
	void tpcPut(String the_key, String the_value) throws RemoteException;
	
	/**
	 * This is the delete request to be replicated on the replicate servers. 
	 * This is the first message associated with the first phase in 2PC.
	 * @param the_request_id - this is the id associated with this request
	 * @param the_request - this is the delete request in string form
	 * @return
	 */
	void tpcDelete(String the_key) throws RemoteException;
	
	/**
	 * This is the GO message in the second phase of the 2PC. This is
	 * is method on remote objects 
	 * @param the_request_id - this is the id for the associated rpc to commit
	 * @return - the ACK message
	 */
	boolean tpcGO(String the_request_id) throws RemoteException;
	
	/**
	 * When the server first comes on line it should find a registry check to 
	 * see if it has the latest version. If not, the method returns a NACK and
	 * callee's remote objects will call the caller's update method sending the
	 * latest version of the replicated KVStore.
	 * @param the_version - the current version of the local KVStore
	 * @return
	 */
	String tpcIsFresh(String the_version) throws RemoteException;
	
	/**
	 * This is called by other server's to update an out-of-date KVStore
	 * @param the_KV_store
	 * @return
	 */
	String tpcUpdate(KVStore the_KV_store) throws RemoteException;
	
	/**
	 * This is called by other server's to update an out-of-date KVStore
	 * @param the_request_id - this is the id for the associated rpc to commit
	 * @param the_request - this is the delete request in string form
	 * @return
	 */
	boolean tpcRequest(String the_request_id, String...the_request) throws RemoteException;
	
	
}