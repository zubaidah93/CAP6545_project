%normalize the data using fro

%the input to this is the subjects individually! 
% input is 24 x 8 x 1280

function z= fnormal(x)
s =0;
for i = 9:32
    m = (squeeze(x(i,:,:)));
    n = norm(m,'fro');
    s = s+n;
    s;
end
z = sqrt(s/(24*8*1280));
end



