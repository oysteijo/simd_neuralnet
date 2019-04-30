/* strtools.h - Øystein Schønning-Johansen 2012
 *
 * Some naive tools for string operations in C.
 */

#ifndef __STRTOOLS_H__
#define __STRTOOLS_H__

#include <stdio.h>
#include <stdlib.h>
#include <stdarg.h>
#include <string.h>
#include <ctype.h>

#define STRSPLIT_INIT \
static char **strsplit( char *in, char delimiter ) \
{ \
	unsigned int i = 0, count = 0; \
	char *string = in; \
	char *pstr; \
	char **strv; \
 \
	for ( pstr = string; *pstr != '\0'; pstr++ ) \
		if( *pstr == delimiter ) count++; \
 \
	strv = (char **) malloc( (2 + count) * sizeof ( char* )); \
	if ( !strv ) return NULL;	 \
	strv[ i++ ] = string; \
 \
	for ( pstr = string; *pstr != '\0'; pstr++ ){ \
		if( *pstr == delimiter ) { \
			*pstr = '\0'; \
			strv[ i++ ] = pstr + 1; \
		} \
	} \
	strv[ i ] = NULL; \
 \
	return strv; \
}

#define STRSPLIT_LENGTH_INIT \
static int strsplit_length( char **strv ) \
{ \
	int count = 0; \
	while ( *strv++ != NULL ) count++; \
	return count; \
} 

#define STRSPLIT_FREE_INIT  \
static void strsplit_free( char **strv ) \
{ \
	free( strv );  \
} 

#define STRCONCAT_INIT \
static char *strconcat( const char *first, ... ) \
{ \
	va_list argptr; \
	char *str; \
	char *retstr; \
	size_t size = strlen(first); \
 \
	va_start( argptr, first ); \
	str = va_arg( argptr, char*); \
	while ( str ){ \
		size += strlen( str ); \
		str = va_arg( argptr, char*); \
	} \
 \
	retstr = (char*) malloc( size * sizeof(char) + 1); \
	if( retstr == NULL ){ \
		perror("Can't allocate memory for 'retstr'.\n"); \
		return NULL; \
	} \
 \
	retstr += sprintf( retstr, "%s", first ); \
 \
	va_start( argptr, first ); \
	while ((str = va_arg( argptr, char* ))) \
		retstr += sprintf( retstr, "%s", str ); \
 \
	va_end( argptr ); \
 \
	retstr[size-1] = '\0'; \
	return retstr; \
} 

#define STRJOIN_INIT  \
static char *strjoin( char separator, ... ) \
{ \
	va_list argptr; \
	char *str; \
	char *retstr; \
	size_t size = 0; \
 \
	va_start( argptr, separator ); \
	str = va_arg( argptr, char*); \
	while ( str ){ \
		size += strlen( str ) + 1; \
		str = va_arg( argptr, char*); \
	} \
 \
	retstr = (char*) malloc( size * sizeof(char)); \
	if( retstr == NULL ){ \
		perror("Can't allocate memory for 'retstr'.\n"); \
		return NULL; \
	} \
 \
	va_start( argptr, separator ); \
	while ((str = va_arg( argptr, char* ))) \
		retstr += sprintf( retstr, "%s%c", str, separator ); \
 \
	va_end( argptr ); \
 \
	retstr[size-1] = '\0'; \
	return retstr; \
} 

#if 0
/* I don't think this works properly. It should return the string allocated -- retstr is moved.*/
char *strjoinv( const char **strv, char separator )
{
	const char **iter = strv;
	char *retstr;
	size_t size = 0;
	while ( *iter ){
		size += strlen( *iter ) + 1;
		iter++;
	}
	retstr = (char*) malloc( size * sizeof(char));
	
	for ( iter = strv; *iter; iter++ )
		retstr += sprintf( retstr, "%s%c", *iter, separator );

	retstr[size-1] = '\0';
	return retstr;
}
#endif

/**
  @brief Remove heading and trailing whitespaces if a string.
  @param str The string to strip
  @return pointer to the srtipped string.
  */
static inline char *strstrip(char *str)
{
	size_t size;
	char *end, *start;

	if( !str ) return NULL;

	size = strlen(str);

	if (!size) return str;

	end = str + size - 1;
	while (end >= str && isspace(*end))
		end--;
	*(end + 1) = '\0';

	start = str;
	while ( isspace(*start ) )
		start++;

	return (char*) memmove( str, start, end - start + 2);
}

#endif /* __STRTOOLS_H_ */
