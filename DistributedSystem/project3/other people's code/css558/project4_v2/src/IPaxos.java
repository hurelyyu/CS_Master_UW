import java.rmi.Remote;
import java.rmi.RemoteException;


public interface IPaxos extends Remote {
	

//	/**
//	 * An other server will call this method on a remote object, passing in its
//	 * own server id.   If the server id is higher that the local host's, then
//	 * return ACK, NACK otherwise.
//	 * @param server_id
//	 * @return
//	 */
//	String checkForLeader(int server_id);
	
	/**
	 *This method receives the slot number that the proposer is trying
	 *pass a proposal for.  If it already accepted a proposal for that slot,
	 *then it returns that proposal to the proposer.  If the proposer, has not
	 *accepted anything for this slot, then this acceptor returns an ACK
	 *message.
	 * @param n - is the proposals index number
	 * @return - the accepted value, o.w. return ACK
	 * @throws InterruptedException 
	 */

	
	public String prepare(float n) throws RemoteException, InterruptedException;
	

	/**
	 * The proposer calls accept on remote objects, passing the index slot 
	 * number and the request to be accepted.  
	 * @param n - index slot number
	 * @param value - the command to be accepted
	 * @return - the index of the first unchosen command or empty slot that is 
	 * 				less than n, o.w. return ACK
	 */
	public String accept(float n, String value) throws RemoteException;
//	
//	/**
//	 * When the proposal receives an index from an acceptor for its first
//	 * unknown chosen array from the result of the accept method or success
//	 * method, it calls this method on the remote object, passing in the 
//	 * chosen command and the commands index in the replicated log.
//	 * @param index
//	 * @param value
//	 * @return
//	 */
//	String success(int index, String value);

	/**
	 * When the proposal receives an index from an acceptor for its first
	 * unknown chosen array from the result of the accept method or success
	 * method, it calls this method on the remote object, passing in the 
	 * chosen command and the commands index in the replicated log.
	 * @param index
	 * @param value
	 * @return
	 */
	void learn(String accepted_propsal_and_value)throws RemoteException;
	
	public int getServer_id() throws RemoteException;
	
}
