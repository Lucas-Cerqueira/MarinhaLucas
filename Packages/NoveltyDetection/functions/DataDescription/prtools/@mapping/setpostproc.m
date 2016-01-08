%SETPOSTPROC (Re)set POSTPROC field of a datafile
%
%   A = SETPOSTPROC(A,MAPPING)
%   A = SETPOSTPROC(A)
%
% INPUT
%   A        - Datafile
%   POSTPROC - Postprocessing mapping command
%
% OUTPUT
%   A       - Datafile
%
% DESCRIPTION
% Sets the mappings stored in A.POSTPROC. The size of the datafile
% A is set to the output size of MAPPING.
% A call without MAPPING clears A.POSTPROC. The size of the datafile
% A is reset to undefined (0).
%
% The mappings in A.POSTPROC may be extended by ADDPOSTPROC.
%
% Mappings in A.POSTPROC are stored only and executed just 
% after A is converted from a DATAFILE into a DATASET.
%
% SEE ALSO
% DATAFILES, SETPREPROC, ADDPOSTPROC.
