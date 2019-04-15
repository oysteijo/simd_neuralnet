/* activation.h - Øystein Schønning-Johansen 2013 */
#ifndef __ACTIVATION_H__
#define __ACTIVATION_H__

typedef void (*activation_func)(const unsigned int n, float *ar );
activation_func get_activation_func( const char * name );
const char * get_activation_name( activation_func prt );

#endif /* __ACTIVATION_H__ */
