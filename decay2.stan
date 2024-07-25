
data {
    int<lower=1> N_widgets;                 // number of widgets
    vector[N_widgets] t_initial;            // Time of manufacture.
    vector[N_widgets] t_measured;           // Time of measurement.
    vector[N_widgets] N_measured;           // Amount of material measured.
    vector[N_widgets] sigma_N_measured;     // Uncertainty in amount measured.
    real N_initial_max;                     // Maximum amount of initial material.
    array[N_widgets] int <lower=1, upper=3> detector;     //detector used to make the reading.
}

parameters {
    real<lower=0> alpha;                    // The decay rate parameter.
    vector[3] detector_bias;                // The bias recorded by each widget.
    
    // The amount of initial material is not known.
    vector<lower=0, upper=N_initial_max>[N_widgets] N_initial;
}

model {
    detector_bias ~ normal(0,1);
    for (i in 1:N_widgets) {
        N_measured[i] ~ normal(N_initial[i] * exp(-alpha * (t_measured[i] - t_initial[i]))
         + detector_bias[detector[i]], sigma_N_measured[i]);
    }
}

