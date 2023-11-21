/* loss.h - Øystein Schønning-Johansen 2013 - 2023 */
/*
  vim: ts=4 sw=4 softtabstop=4 expandtab 
 */
#ifndef __LOSS_H__
#define __LOSS_H__

typedef void (*loss_func)(unsigned int n, const float *y_pred, const float *y_real, float *loss );
loss_func get_loss_func( const char * name );
const char * get_loss_name( loss_func ptr );

#endif /* __LOSS_H__ */
