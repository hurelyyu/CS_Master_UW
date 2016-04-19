/**
 * Created by binrusas on 4/8/14.
 */
import java.io.*;
import java.net.*;

public class UDPClient{
    private InetAddress serverIPAddress;
    private int serverPort;
    private DatagramSocket clientSocket;
    private Logger logger ;
    private final int SECOOND = 1000;

    public UDPClient(String server, int serverPort) throws IOException {
        this.serverPort = serverPort;
        this.serverIPAddress = InetAddress.getByName(server);
        clientSocket = new DatagramSocket();
        logger = new Logger("UDP-client-log.txt");
        logger.log("Client is running on : " + Inet4Address.getLocalHost());
        System.out.println("Client is running on : " + Inet4Address.getLocalHost());
    }

    private void put(String key, String value) throws IOException {
        String request = "PUT," + key + "," + value;
        handleRequest(request);
    }

    public void closeSocket(){
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

    public String getRidOfAnnoyingChar(DatagramPacket packet){
        String result = new String(packet.getData());
        char[] annoyingchar = new char[1];
        char[] charresult = result.toCharArray();
        result = "";
        for(int i=0;i<charresult.length;i++){
            if(charresult[i]==annoyingchar[0]){
                break;
            }
            result+=charresult[i];
        }
        return result;
    }

    private void handleRequest(String request) throws IOException {
        byte[] sendData = request.getBytes();
        DatagramPacket sendPacket = new DatagramPacket(sendData, sendData.length, serverIPAddress, serverPort);
        clientSocket.send(sendPacket);
        logger.log("Client has send request:(" + request + ") to server = (" + serverIPAddress + ") " +
                "on port = (" + serverPort + ")");
        System.out.println("Client has send request:(" + request + ") to server = (" + serverIPAddress + ") " +
                "on port = (" + serverPort + ")");

        byte[] receiveData = new byte[1024];
        DatagramPacket receivePacket = new DatagramPacket(receiveData, receiveData.length);

        clientSocket.setSoTimeout(SECOOND * 10);   // set the timeout to 10 seconds
        int trials = 0 ;
        boolean received = false;
        while(!received && trials < 10){
            try{
                clientSocket.receive(receivePacket);
                String response = getRidOfAnnoyingChar(receivePacket);
                logger.log("Client has receive response from server (" + receivePacket.getAddress() + ") ," +
                        " port: " + receivePacket.getPort() + "\nThe response message is : (" + response + ")");
                System.out.println("Client has receive response from server (" + receivePacket.getAddress() + ") ," +
                        " port: " + receivePacket.getPort() + "\nThe response message is : (" + response + ")");
                received = true;
            } catch (SocketTimeoutException e) {
                // resend
                clientSocket.send(sendPacket);
                logger.log("Timeout reached");
                System.out.println("Timeout reached");
                logger.log("Re-send = (" + request + ") has been issued to server = (" + serverIPAddress + ") " +
                        "on port = (" + serverPort + ")");
                System.out.println("Re-send = (" + request + ") has been issued to server = (" + serverIPAddress + ") " +
                        "on port = (" + serverPort + ")");
                trials++;
                continue;
            }
        }
    }

    public static void main(String args[]) throws Exception
    {
        if(args.length < 2){
            System.out.println("Error: no enough parameters");
            System.exit(2);
        }

        String server = args[0];
        int port = Integer.parseInt(args[1]);
        UDPClient c = new UDPClient(server, port);
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
        c.closeSocket();
        c.closeLogger();
    }
}

