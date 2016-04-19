Yaqun Yu and Xiao Liang 558 Project2 ReadMe
********************
For local two users:
********************

Open Server terminal on one computer:
1. Get IP address
2. go to the project Server folder directory(use cd command)
3. $ copy server to 5 different VMs: n01, n02, n03, n04, n05
4. $ make
5. $ rmiregistry & on each VM
6. on n01: $ java RMIserver n02 n03 n04 n05
   on n02: $ java RMIserver n01 n03 n04 n05
   on n03: $ java RMIserver n01 n02 n04 n05
   on n04: $ java RMIserver n01 n02 n03 n05
   on n05: $ java RMIserver n01 n02 n03 n04
	
	
open Client terminal on another computer:
1. go to the project Client folder directory(use cd command)
2. $ copy client to 2 different VMs: n06, n07
3. $ make
4. on n06: $ java RMIclient <Server IP Address that you want to call>
   on n07: $ java RMIclient <Server IP Address that you want to call>


To observe atomicity in our two phase commit procedure, you can kill any one of
the servers with the ctrl-c command.

************************
For VM on school server:
************************

1. get into 558 project VM n01, use as client, use $ mkdir Client command create Client folder
2. put Client files on /home/NetID/Client/
3. open another VM n02, use as server, use $ mkdir Server command create Server folder
4. put Server files on /home/NetID/Server/ through VM n01
5. repeat local instruction for each side.
6. command to check file: $ vi <File full name, ex: RMIServerLog.txt>


*******************ATTENTION!!!***************
1. In order to test exception can also works, we put some error test situation into our test set on purpose. 
2. In case our txt file change format, we print out a pdf version in our document as well
3. Default Server port would be 1099
4. $ ifconfig command can get ip address
5. if communication fail or IP address is incorrect, Exception will throw and write in log