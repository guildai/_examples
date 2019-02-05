check:
	@for path in `find */Makefile`; do \
	  make -C $$(dirname $$path) check; \
	done
