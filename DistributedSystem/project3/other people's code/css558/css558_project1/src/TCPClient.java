/**
 * Created by binrusas on 4/9/14.
 */

import java.io.BufferedReader;
import java.io.DataOutputStream;
import java.io.IOException;
import java.io.InputStreamReader;
import java.net.Inet4Address;
import java.net.InetAddress;
import java.net.Socket;

public class TCPClient {
    private InetAddress serverIPAddress;
    private int serverPort;
    private Socket clientSocket;
    private Logger logger ;

    public TCPClient(String server, int serverPort) throws IOException {
        this.serverPort = serverPort;
        this.serverIPAddress = InetAddress.getByName(server);
        clientSocket = new Socket(serverIPAddress, serverPort);
        logger = new Logger("TCP-client-log.txt");
        System.out.println("TCP-client-log.txt");
        logger.log("Client is running on : " + Inet4Address.getLocalHost());
        System.out.println("Client is running on : " + Inet4Address.getLocalHost());
    }

    private void put(String key, String value) throws IOException {
        String request = "PUT," + key + "," + value;
        handleRequest(request);
    }

    public void closeSocket() throws IOException {
        clientSocket.close();
    }

    public void closeLogger(){
        logger.close();
    }

    private void get(String key) throws IOException {
        String request = "GET," + key;
        handleRequest(request);
    }

    private void delete(String key) throws IOException {
        String request = "DELETE," + key;
        handleRequest(request);
    }

    private void handleRequest(String request) throws IOException {
        DataOutputStream outToServer = new DataOutputStream(clientSocket.getOutputStream());
        BufferedReader inFromServer = new BufferedReader(new InputStreamReader(clientSocket.getInputStream()));
        outToServer.writeBytes(request + '\n');
        logger.log("Client has send request:(" + request + ") to server = (" + serverIPAddress + ") " +
                "on port = (" + serverPort + ")");
        System.out.println("Client has send request:(" + request + ") to server = (" + serverIPAddress + ") " +
                "on port = (" + serverPort + ")");

        String response = inFromServer.readLine();

        logger.log("Client has receive response from server (" + serverIPAddress + ") , " +
                "port: " + serverPort + "\nThe response message is : (" + response + ")");
        System.out.println("Client has receive response from server (" + serverIPAddress + ") , " +
                "port: " + serverPort + "\nThe response message is : (" + response + ")");

    }

    public static void main(String args[]) throws Exception
    {
        if(args.length < 2){
            System.out.println("Error: not enough parameters");
            System.exit(2);
        }

        String server = args[0];
        int port = Integer.parseInt(args[1]);
        TCPClient c = new TCPClient(server, port);
//        client.put("red" , "R");
//        client.put("green" , "G");
//        client.put("blue" , "B");
//        client.put("black" , "BLC");
//        client.put("yellow" , "Y");
//
//        client.get("red");
//        client.delete("red");
//        client.get("red");
		c.put("key1", "value1");
		c.put("key2", "value2");
		c.put("key3", "value3");
		c.put("key4", "value4");
		c.put("key5", "value5");
		c.get("key1");
		c.get("key2");
		c.get("key3");
		c.get("key4");
		c.get("key5");
		c.delete("key1");
		c.delete("key2");
		c.delete("key3");
		c.delete("key4");
		c.delete("key5");
		c.get("key5");
        c.closeSocket();
        c.closeLogger();
    }
}


