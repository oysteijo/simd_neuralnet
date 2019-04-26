/* metric.h - Øystein Schønning-Johansen 2013 */
#ifndef __METRICS_H__
#define __METRICS_H__

typedef void (*metric_func)(unsigned int n, const float *y_pred, const float *y_real);
metric_func get_metric_func( const char * name );
const char * get_metric_name( metric_func ptr );

#endif /* __METRICS_H__ */
