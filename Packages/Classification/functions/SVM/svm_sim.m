function [ svm_output, output] = svm_sim( svm_struct, data )
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here

sv = svm_struct.SupportVectors;
alphaHat = svm_struct.Alpha;
bias = svm_struct.Bias;
kfun = svm_struct.KernelFunction;
kfunargs = svm_struct.KernelFunctionArgs;

f = (feval(kfun,sv,data,kfunargs{:})'*alphaHat(:)) + bias;

out = sign(f);
% points on the boundary are assigned to class 1
out(out==0) = 1;

svm_output = out;
output = f;

end

