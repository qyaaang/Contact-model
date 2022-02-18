function [obj_val] = optim_obj(params)
%Optim_obj Summary of this function goes here
%   Detailed explanation goes here
    obj_1 = py.importlib.import_module('optim_args');
    py.importlib.reload(obj_1);
    obj_2 = py.importlib.import_module('optim_obj');
    py.importlib.reload(obj_2);
    args = py.optim_args.input_args();
    optim = py.optim_obj.OptimObj(args);
    obj_val = optim.objective(params);
end
