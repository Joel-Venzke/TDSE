
all: TDSE

TDSE:
	$(MAKE) -C src

clean:
	$(MAKE) clean -C src
