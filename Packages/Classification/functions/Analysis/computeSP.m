function [SP,eff_per_class] = computeSP(target,output)
%COMPUTESP Summary of this function goes here
%   Detailed explanation goes here


warning off;
SP = [];

[~,~,~,d] = confusion(target,output);
eff_per_class = d(:,3);

arit_mean = sum(eff_per_class)/length(eff_per_class);
geo_mean = prod(eff_per_class)^(1/length(eff_per_class));
SP = sqrt(arit_mean*geo_mean);
warning on;

% n_thr = 25;
% possible_thr = linspace(-1.1,1,n_thr);
% 
% for i_thr = 1:size(possible_thr,2)
%     new_output = output;
%     new_output(new_output < possible_thr(i_thr)) = 0;
%     new_output(new_output >= possible_thr(i_thr)) = 1;
%     
%     new_target = target;
%     new_target(new_target < 0) = 0;
%     new_target(new_target >= 0) = 1;
%     
%     [~,~,~,d] = confusion(new_target,new_output);
%     
%     eff_per_class = d(:,3);
% 
%     arit_mean = sum(eff_per_class)/length(eff_per_class);
%     geo_mean = prod(eff_per_class)^(1/length(eff_per_class));
% 
%     SP(i_thr) = sqrt(arit_mean*geo_mean);
% end
% 
% %fprintf('Max SP: %1.3f\n',max(SP));
% 
% id_max_SP = find(SP == max(SP));
% id_max_SP = id_max_SP(1);
% 
% new_output = output;
% new_output(new_output < possible_thr(id_max_SP)) = 0;
% new_output(new_output >= possible_thr(id_max_SP)) = 1;
% 
% new_target = target;
% new_target(new_target < 0) = 0;
% new_target(new_target >= 0) = 1;
% 
% [~,~,~,d] = confusion(new_target,new_output);
% 
% eff_per_class = d(:,3);
% arit_mean = sum(eff_per_class)/length(eff_per_class);
% geo_mean = prod(eff_per_class)^(1/length(eff_per_class));
% 
% SP = sqrt(arit_mean*geo_mean);

end

