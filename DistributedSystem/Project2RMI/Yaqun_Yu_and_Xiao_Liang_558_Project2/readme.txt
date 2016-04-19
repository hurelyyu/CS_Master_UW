Yaqun Yu and Xiao Liang 558 Project2 ReadMe
********************
For local two users:
********************

Open Server terminal on one computer:
1. Get IP address
2. go to the project Server folder directory(use cd command)
3. $ make
4. $ rmiregistry &
5. $ java RMIserver <port number>

open Client terminal on another computer:
1. go to the project Client folder directory(use cd command)
2. $ make
3. $ java RMIclient2 <Server IP Address>

and Wala!
the result will show up

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