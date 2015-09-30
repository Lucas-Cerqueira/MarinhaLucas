function [nn, trn_desc] = train_neural_network(input,target,itrain,ivalid,itest,top,train_fnc,perf_fnc,act_fnc,n_epochs,show)
%TRAIN_NEURAL_NETWORK Summary of this function goes here
%   Detailed explanation goes here

if show,
    fprintf('\n \nFunction train_neural_network\n');
    fprintf('nargin: %i\n', nargin);
    fprintf('Size Input: %i lin %i col\n', size(input,1), size(input,2));
    fprintf('Size Target: %i lin %i col\n', size(target,1), size(target,2));
    fprintf('Size iTrain: %i lin %i col\n', size(itrain,1), size(itrain,2));
    fprintf('Size iValid: %i lin %i col\n', size(ivalid,1), size(ivalid,2));
    fprintf('Size iTest: %i lin %i col\n', size(itest,1), size(itest,2));
    fprintf('Top: %i\n', top);
    fprintf('Train Fnc: %s\n', train_fnc);
    fprintf('perf_fnc: %s\n', perf_fnc);
    fprintf('act_fnc: %s\n', act_fnc{1});
    fprintf('Epochs: %i\n', n_epochs);
end

net = newff(input,target,top,act_fnc,train_fnc);
net = revert(net);
net = init(net);

%net.iw{1,1}, net.b{1}

%Definicoes de Treinamento.
net.trainParam.lr               = 0.000001;
net.trainParam.max_fail         = 150;
net.trainParam.mc               = 0.99999;
net.trainParam.min_grad         = 1e-10;
net.trainParam.goal             = 0;
net.trainParam.epochs           = n_epochs;
net.performFcn                  = perf_fnc;
net.trainParam.show             = net.trainParam.max_fail;
net.trainParam.showWindow       = false;
net.trainParam.showCommandLine  = show;

net.divideFcn                   = 'divideind';
net.divideParam.trainInd        = itrain;
net.divideParam.testInd         = itest;
net.divideParam.valInd          = ivalid;

%Treinamento da rede.
[nn,trn_desc]   = train(net,input,target);

%net.iw{1,1}, net.b{1}

end

