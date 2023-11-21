/* metrics.h - Øystein Schønning-Johansen 2019 - 2023 */
/*
  vim: ts=4 sw=4 softtabstop=4 expandtab 
 */
#ifndef __METRICS_H__
#define __METRICS_H__

typedef float (*metric_func)(const int n, const float *y_pred, const float *y_real );
metric_func get_metric_func( const char * name );
const char * get_metric_name( metric_func ptr );

#endif /* __METRICS_H__ */
