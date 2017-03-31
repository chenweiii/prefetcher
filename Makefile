CFLAGS = -msse2 -mavx -mavx2 --std gnu99 -g -O0 -Wall -Wextra

GIT_HOOKS := .git/hooks/applied

all: $(GIT_HOOKS) main.c
	$(CC) $(CFLAGS) -DSSE_PREFETCH -o sse_prefetch main.c
	$(CC) $(CFLAGS) -DSSE -o sse main.c
	$(CC) $(CFLAGS) -DNAIVE -o naive main.c
	$(CC) $(CFLAGS) -DAVX -o avx main.c

$(GIT_HOOKS):
	@scripts/install-git-hooks
	@echo

clean:
	$(RM) sse sse_prefetch naive avx
