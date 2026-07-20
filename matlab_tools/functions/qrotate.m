function  [v_new] = qrotate(v0,q)
% Function to rotate a vector v0 by a quaternion q 
% INPUT: 
%    v0 = array [3x1],vector
%    q = Quaternion   
% OUTPUT:
%   v_new = array[3x1], vector

    % Transfrom vect into an quaternion 
    v1 = horzcat(0,v0); % add 0 to scalar postion to make it a quaternion
    % Normalize it
    norm_vect = norm(v1);
%     v1 = v1/norm(v1);
    % Conjugate of quaternion
    q_conjugate = [q(1),-q(2),-q(3),-q(4)];
    
    % The result is given by: quat * vect * quat_conjugate
%     vector_new = quat_mult(q,quat_mult(v1,q_conjugate))* norm_vect;
%     Commented KP
%     vector_new = quat_mult(q,quat_mult(v1,q_conjugate));
    product_0 = quat_mult(q,v1);
    vector_new = quat_mult(product_0, q_conjugate);
    % Slice it to get back in vector form
    [v_new] = vector_new(2:end)';
end
function r = quat_mult(q,p)
% Quaternion multiplication. 
% KP 28-04-2023 REVERSED inputs from (p,q) to (q,p)
% unit tests and Sources (Crassidis 2014, Matlab's qrotate page)
% this is the sequence for a Hamiltonian product.
% if the original is used, then it must be q_conj V q
% for this reversed input - q V q_conj is verified to work!
% INPUT: 
%    Quaternion p
%    Quaternion q
% OUTPUT:
%    Quternion, r = p * q

r =  [  p(1) * q(1) - p(2) * q(2) - p(3) * q(3) - p(4) * q(4),
        p(1) * q(2) + p(2) * q(1) + p(3) * q(4) - p(4) * q(3),
        p(1) * q(3) - p(2) * q(4) + p(3) * q(1) + p(4) * q(2),
        p(1) * q(4) + p(2) * q(3) - p(3) * q(2) + p(4) * q(1)];

end