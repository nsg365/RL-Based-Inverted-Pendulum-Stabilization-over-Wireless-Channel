function u = lqr_controller(x, K, p)

u = -K * x;
u = max(min(u, p.umax), -p.umax);

end
