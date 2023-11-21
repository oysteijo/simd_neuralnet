/* activation.h - Øystein Schønning-Johansen 2013 - 2023 */
/*
  vim: ts=4 sw=4 softtabstop=4 expandtab 
 */
#ifndef __ACTIVATION_H__
#define __ACTIVATION_H__

typedef void (*activation_func)      (const int n, float *ar );
typedef void (*activation_derivative)(const int n, const float *activation, float *ar );

activation_func       get_activation_func      ( const char * name );
activation_derivative get_activation_derivative( const activation_func ptr );
const char *          get_activation_name      ( const activation_func ptr );

#endif /* __ACTIVATION_H__ */
