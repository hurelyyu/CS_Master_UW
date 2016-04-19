JFLAGS = -g
JC = javac
.SUFFIXES: .java .class
.java.class:
	$(JC) $(JFLAGS) $*.java

CLASSES = \
	myHashMap.java \
	myLog.java \
	TCPClientMain.java \
	TCPServerMain.java \
	UDPServerMain.java \
	UDPClientMain.java \
	testMain.java  
	
default: classes

classes: $(CLASSES:.java=.class)

clean:
	$(RM) *.class
