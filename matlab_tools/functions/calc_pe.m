function pe_urad = calc_pe(vec_1,vec_2)
%CALC_PE Calculate angle between two vectors
% Output in mrad
pe = acos(dot(vec_1, vec_2) / (norm(vec_1) * norm(vec_2)));
pe_urad = pe * 1e3;

end

