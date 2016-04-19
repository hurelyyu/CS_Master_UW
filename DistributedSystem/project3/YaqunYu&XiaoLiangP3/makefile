JFLAGS = -g
JC = javac
.SUFFIXES: .java .class
.java.class:
	$(JC) $(JFLAGS) $*.java

CLASSES = \
	RMIinterface.java \
	RMIclient.java \
	myLog.java \
	RMI_two_phases_commit.java \
	RMIserver.java \
	RMIServerStoreData.java \
	

default: classes

classes: $(CLASSES:.java=.class)

clean:
	$(RM) *.class
