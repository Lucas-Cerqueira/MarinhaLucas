% DD_MEANPREC compute the mean precision
%
%      MAP = DD_MEANPREC(E)
%
% INPUT
%   E     Precision-recall graph
%
% OUTPUT
%   MAP   Mean average precision
%
% DESCRIPTION
% Compute the mean precision from a precision-recall curve (as obtained
% by dd_prc).
%
% SEE ALSO
% dd_error, dd_prc, dd_auc.
%

% Copyright: D.M.J. Tax, D.M.J.Tax@prtools.org
% Faculty EWI, Delft University of Technology
% P.O. Box 5031, 2600 GA Delft, The Netherlands

function err = dd_meanprec(e)
% First check if we are dealing with an PrecRecall structure as it is
% delivered by dd_prc:
if isa(e,'struct')
	if ~isfield(e,'type')
		error('The curve should be a precision-recall curve (by dd_prc).');
	end
	if ~strcmp(e.type,'prc')
		error('The curve should be a precision-recall curve, no ROC curve.');
	end
	if isfield(e,'err')
		e = e.err;
	else
		error('E seems to be a struct, but does not contain an E.err field');
	end
else
	
	%error('Please supply a valid precision-recall curve (by dd_prc)');
end

% first sort/flip it??
% OK this does not work: when we have equal values for e(:,2), they are
% not flipped in order:
%[newrec,I] = sort(e(:,2)); newprec = e(I,1);
if e(1,2)>e(end,2),
	newrec = flipud(e(:,2));
	newprec = flipud(e(:,1));
else
	newrec = e(:,2);
	newprec = e(:,1);
end
% add the first point... magic!: copy the first element from newprec:
newrec = [0; newrec];
newprec = [newprec(1); newprec];
% from the VOC competition:
n = size(e,1);
mxprec = newprec(n);
for i=n:-1:1
	if newprec(i)>mxprec
		mxprec = newprec(i);
	else
		newprec(i) = mxprec;
	end
end

% area under the curve:
drec = diff(newrec);
dprec = newprec(2:end);
err = drec'*dprec;

return
