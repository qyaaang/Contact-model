clc
clear
fun = @optim_obj;
lb = [0.5 500 500 0.1];
ub = [2.0 1200 1200 0.5];
x = ga(fun,4,[],[],[],[],lb,ub);