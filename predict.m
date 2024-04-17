% prediction function

function t = predict(x, class_1_mean, class_2_mean, class_3_mean)
d1 = distance_riemann(class_1_mean, x);
d2 = distance_riemann(class_2_mean, x);
d3 = distance_riemann(class_3_mean, x);
%d4 = distance_riemann(class_rest_mean, x);
pred_label = min([d1 d2 d3]);
if pred_label == d1
    t = 13;
elseif pred_label == d2
    t = 17;
elseif pred_label == d3
    t = 21;
% elseif pred_label == d4
%     t = 0;
end
end