JFLAGS = -g
JC = javac
.SUFFIXES: .java .class
.java.class:
	$(JC) $(JFLAGS) $*.java

CLASSES = \
	RMIinterface.java \
	RMIclient2.java \
	myLog.java \
	

default: classes

classes: $(CLASSES:.java=.class)

clean:
	$(RM) *.class
