#include <stdio.h>
#include <unistd.h> 
#if __WIN32
#include <windows.h>
#define EXTRA_WIDTH 8
#define isatty _isatty
#else /* (Linux?) */
#include <sys/ioctl.h>
#define EXTRA_WIDTH 7
#endif

/* gets the current screen column width */
static unsigned short getcols(int fd)
{
	const unsigned short default_tty = 80;
	unsigned short termwidth = 0;
	const unsigned short default_notty = 0;
	if(!isatty(fd)) {
		return default_notty;
	}
#if __WIN32
    HANDLE console;
    CONSOLE_SCREEN_BUFFER_INFO info;
    /* Create a handle to the console screen. */
    console = CreateFileW(L"CONOUT$", GENERIC_READ | GENERIC_WRITE,
            FILE_SHARE_READ | FILE_SHARE_WRITE, NULL, OPEN_EXISTING,
            0, NULL);
    if (console == INVALID_HANDLE_VALUE)
        return default_tty;

    /* Calculate the size of the console window. */
    if (GetConsoleScreenBufferInfo(console, &info) == 0)
        return default_tty;
    CloseHandle(console);
    termwidth = info.srWindow.Right - info.srWindow.Left + 1;
#else  /* Not Windows */

#if defined(TIOCGSIZE)
	struct ttysize win;
	if(ioctl(fd, TIOCGSIZE, &win) == 0) {
		termwidth = win.ts_cols;
	}
#elif defined(TIOCGWINSZ)
	struct winsize win;
	if(ioctl(fd, TIOCGWINSZ, &win) == 0) {
		termwidth = win.ws_col;
	}
#endif
#endif  /* Not Windows */
	return termwidth == 0 ? default_tty : termwidth;
}

static inline void progress_bar( const char *label, int x, int n )
{
	if ( (x != n) && (x % (n/100 + 1) != 0) ) return;

	int w = getcols( STDOUT_FILENO ) - EXTRA_WIDTH - strlen( label );
	if( w < 1 ) return;

	// Calculuate the ratio of complete-to-incomplete.
	float ratio = x/(float)n;
	int   c     = ratio * w;

	printf("%s[", label );

	for (int i=0; i<c; i++) printf("#");

    for (int i=c; i<w; i++) printf("-");

	// ANSI Control codes to go back to the
	// previous line and clear it.
	printf("] %3d%%", (int)(ratio*100));
	printf( x == n ? "\n" : "\r");
	fflush( stdout );
}

