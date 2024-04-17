

function x = BP(f1, f2, f3, z)
fs = 128;
Fn = fs/2;
if nargin == 2
    n_butter = 8;
    [b,a]= butter(n_butter, f1./Fn, 'bandpass');
    x = filtfilt(b, a, f2); % band pass
elseif nargin == 4
    Rp = 3;%passband ripple
    Rs = 10;%stopband ripple
    Wp1 = f1/Fn; Ws1 = [f1(1)-1 f1(2)+1]/Fn; %first passband freq
    Wp2 = f2/Fn; Ws2 = [f2(1)-1 f2(2)+1]/Fn; %second pbf
    Wp3 = f3/Fn; Ws3 = [f3(1)-1 f3(2)+1]/Fn; %third pbf
   
    
    %find the cutoff frequency and the order of the filter
    [n1,Wn1] = buttord(Wp1,Ws1,Rp,Rs);
    [n2,Wn2] = buttord(Wp2,Ws2,Rp,Rs);
    [n3,Wn3] = buttord(Wp3,Ws3,Rp,Rs);
    
    %find the system function of the filter
    [b1, a1] = butter(n1, Wn1, 'bandpass');
    [b2, a2] = butter(n2, Wn2, 'bandpass');
    [b3, a3] = butter(n3, Wn3, 'bandpass');
    
    x1 = filtfilt(b1, a1, z); %Band pass for f1
    x2 = filtfilt(b2, a2, z); %Band pass for f2
    x3 = filtfilt(b3, a3, z); %Band pass for f3
    
    x = [x1 x2 x3];

end

end


