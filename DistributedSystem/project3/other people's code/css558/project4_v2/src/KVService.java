/*
 *Stephen Mosby
 *Kellen Han-Nin Cheng
 *Aqeel S Bin Rustum
 *Nai-Wei Chen
 *CSS558 Sp14 Project3
*/

import java.rmi.Remote;
import java.rmi.RemoteException;

/**
 * A simple interface, describing the methods needed to be implemented in the
 * server-side code, and called remotely by the client-side code.
 *
 */
public interface KVService  extends Remote {

	/**
	 * This method returns the value specified by the key.
	 * @param the_key the key value used in a KVStore
	 * @return the value mapped to by the given key value
	 * @throws RemoteException
	 */
	String get(String the_key) throws RemoteException;
	
	/**
	 * This method stores the given key-value pair in the KVStore.
	 * @param the_key - the key used in a KVStore
	 * @param the_value - the value stored in a KVStore
	 * @throws RemoteException
	 */
	void put(String the_key, String the_value) throws RemoteException;
	
	/**
	 * This method deletes the value mapped to by the given key.
	 * @param the_key - the key used in a KVStore
	 * @throws RemoteException
	 */
	void delete(String the_key) throws RemoteException;
}