%% LiDAR tracking example
clear all, close all

%% create synthetic data
% two vehicles, constant speed, noise
N = 50;
T = 0.1;
obj1_y0 = 30;
obj1_v0 = -25/3.6;

obj2_y0 = 40;
obj2_v0 = -30/3.6;

obj1_y = obj1_y0 + obj1_v0 * T * (0:N);
obj2_y = obj2_y0 + obj2_v0 * T * (0:N);

obj1_y = obj1_y + 0.2*randn(size(obj1_y));
obj2_y = obj2_y + 0.2*randn(size(obj2_y));

data = cell(N,1);
for k=1:N
    data{k} = [obj1_y(k), obj2_y(k)];
end

%% show data
figure
hold on
for k=1:N
    obs = data{k};
    for j=1:length(obs)
        plot(k, obs(j), 'bo')
    end
end
grid on
hold off

%% tracking
% assume that WIM is passed at k = 12 at distance of 2nd vehicle
k_WIM = 12;
y_WIM = obj2_y0 + obj2_v0 * T * k_WIM; % Position Fzgfront bei WIM Detektion
v_WIM = obj2_v0 + 2/3.6 * randn;

% Kalman
Ad = [
    1 T;
    0 1
];
Gd = [
    T
    1
];
Q = (1/3.6)^2;

C = [
    1 0
];
R = (0.3)^2;

distance_thresh = 1.5;

% Initialzustand
x_dach = [
    y_WIM;
    v_WIM
];
P_dach = [
    (5)^2 0;
    0 (10/3.6)^2    
];

y_filtered = [];
v_filtered = [];
for k=k_WIM:length(data)
    % Daten-Assoziation
    [minValue,closestIndex] = min(abs(data{k}-x_dach(1)));
    obs = [ data{k}(closestIndex) ];
    if abs(data{k}(closestIndex)-x_dach(1))>distance_thresh
        obs = []; % observation too far away from prediction, do not accepte measurement
    end

    % Korrektur
    if ~isempty(obs)
        K = P_dach*C'*pinv(C*P_dach*C' + R);
        x_tilde = x_dach + K*(obs - C*x_dach);
        P_tilde = (eye(length(x_dach)) - K*C)*P_dach;
    end

    % Prädiktion
    x_dach = Ad*x_tilde;
    P_dach = Ad*P_tilde*Ad' + Gd*Q*Gd';

    % logging
    y_filtered = [y_filtered, x_tilde(1)];
    v_filtered = [v_filtered, x_tilde(2)];
end

hold on
plot(k_WIM:N, y_filtered, 'rx-');
hold off

