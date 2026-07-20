in this subfolder, scripts/functions specifically for attitude (quaternion) predictions are stores

generate attitude with main_prop_customer_rates.py -> Euler angles/rates + angular velocity vectors
run attitude rpedictions with main_evaluate_attitude_prediction.py -> Convert to quaternion/rates, slice attitude messages, predict and evaluate PE

pe_att_extrap_xmsg_plot.py-> Plot PE with x_data being number of available att. messages == time [s] * message_upd_rate [N_msg/s]->[N_msg]
