Stephen Mosby
Kellen Han-Nin Cheng
Aqeel S Bin Rustum
Nai-Wei Chen
CSS558 Sp14 Project2


To run the code on the nodes:

1.  Copy the following files to n01:
		-RMIServer.java
		-KVService.java
		-KVStore.java
		-Logger.java
		
2.	Copy the following files to n02:
		-RMIClient.java
		-ClientMain.java
		-KVService.java
		
3.  Run the following commands on n01 with the server files:
		javac RMIServer.java
		java RMIServer

4.	Run the following commands on n02 with the client files:
		javac ClientMain.java
		java ClientMain n01
		
		
Two client threads should now be running n02.