clear all;
clc;
u = input('Insert file name (.wav): ','s');
train(4,5);
test(u,ans);
% https://github.com/yashdv/Speech-Recognition

function code = train(n,a)
% Speaker Recognition: Training Stage

k = 32;      % number of centroids required
for i=1:n          % train a VQ codebook for each speaker
    for I=1:a
    file = sprintf('%ds%d.wav', i, I);           
    disp(file);
   
    [s, fs] = audioread(file);


%      %% butterworth filter
%      N = size(s,1);
%      beginFreq = 7000 / (fs/2);
%     
%      [b,a] = butter(n,beginFreq);
%     
%      s = filter(b, a, s);
     %%
     v = mfcc(s, fs);
     v(isnan(v)) = 0;% Compute MFCC's

    FULLcode{I,i} = vqCodeBook(v, k);      % Train VQ codebook
end
end
for Q=1:n
    Y{Q} = cat(3,FULLcode{:,Q}); % averaging over all 5 samples for each speaker
    code{Q} = mean(Y{1,Q},3);
    
end
end

function c = mfcc(s, fs)
% MFCC Calculate the mel frequencey cepstrum coefficients (MFCC) of a signal
%
% Inputs:
%       s       : speech signal
%       fs      : sample rate in Hz
%
% Outputs:
%       c       : MFCC output, each column contains the MFCC's for one speech frame

N = single(256);                        % frame size
M = 100;                        % inter frame distance
len = length(s);
numberOfFrames = 1 + floor((len - N)/double(M));
mat = zeros(N, numberOfFrames); % vector of frame vectors

for i=1:numberOfFrames
    index = 100*(i-1) + 1;
    for j=1:N
        mat(j,i) = s(index);
        index = index + 1;
    end
end

hamW = hamming(N);              % hamming window
afterWinMat = diag(hamW)*mat;   
freqDomMat = fft(afterWinMat);  % FFT into freq domain

filterBankMat = melFilterBank(20, N, fs);                % matrix for a mel-spaced filterbank
nby2 = 1 + floor(N/2);
ms = filterBankMat*abs(freqDomMat(1:nby2,:)).^2; % mel spectrum
c = dct(log(ms));                                % mel-frequency cepstrum coefficients
c(1,:) = []; 
end% exclude 0'th order cepstral coefficient

function m = melFilterBank(p, n, fs)
% Determine matrix for a mel-spaced filterbank
%
% Inputs:       p   number of filters in filterbank
%               n   length of fft
%               fs  sample rate in Hz
%
% Outputs:      x   a (sparse) matrix containing the filterbank amplitudes
%                   size(x) = [p, 1+floor(n/2)]

f0 = 700 / fs;
fn2 = floor(n/2);

lr = log(1 + 0.5/f0) / (p+1);

% convert to fft bin numbers with 0 for DC term
bl = n * (f0 * (exp([0 1 p p+1] * lr) - 1));

b1 = floor(bl(1)) + 1;
b2 = ceil(bl(2));
b3 = floor(bl(3));
b4 = min(fn2, ceil(bl(4))) - 1;

pf = log(1 + (b1:b4)/n/f0) / lr;
fp = floor(pf);
pm = pf - fp;

r = [fp(b2:b4) 1+fp(1:b3)];
c = [b2:b4 1:b3] + 1;
v = 2 * [1-pm(b2:b4) pm(1:b3)];

m = sparse(double(r), double(c), double(v), double(p), double(1+fn2));
end


function codebk = vqCodeBook(d, k)
% VQLBG Vector quantization using the Linde-Buzo-Gray algorithm
%
% Inputs:
%       d contains training data vectors (one per column)
%       k is number of centroids required
%
% Outputs:
%       c contains the result VQ codebook (k columns, one for each centroids)

e = 0.0001;                                 % splitting parameter
codebk = mean(d, 2);                        % code book
distortion = int32(inf);             
numOfCentroids = int32(log2(k));            % number of code words/centroids

for i=1:numOfCentroids
    codebk = [codebk*(1+e), codebk*(1-e)];  % the splitting
    while(1==1)
        dis = distance(d, codebk);            % distance of each point to every code word
        [m,ind] = min(dis, [], 2);          % ind maps points in 'd' to closest centroid
        t = 0;
        lim = 2^i;
        for j=1:lim
            codebk(:, j) = mean(d(:, ind==j), 2);    % updating centroids to better mean values
            x = distance(d(:, ind==j), codebk(:, j));  % x is a cluster i.e vector of neighbouring ...
            len = length(x);                         % ... points of a centroid
            for q = 1:len
                t = t + x(q);
            end
        end
        if (((distortion - t)/t) < e)       % distortion condition breaks the loop
            break;
        else
            distortion = t;
        end
    end
end
end

function test(u, code)
% Speaker Recognition: Testing Stage
   
    [s, fs] = audioread([u,'.wav']);      
    
    v = mfcc(s, fs);            % Compute MFCC's
   
    distmin = inf;
    k1 = 0;
   
    for l = 1:length(code)      % each trained codebook, compute distortion
        d = distance(v, code{l}); 
        dist = sum(min(d,[],2)) / size(d,1);
      
        if dist < distmin
            distmin = dist;
            k1 = l;
        end      
    end
    if k1 == 0
        msg = ('does not match with the database');
    else
   
    msg = sprintf('matches with speaker %d', k1);
    end
    disp(['Audio file: (',u,'.wav) ',msg])

end

function d = distance(x, y)
% DISTEU Pairwise Euclidean distances between columns of two matrices
%
% Input:
%       x, y:   Two matrices whose each column is an a vector data.
%
% Output:
%       d:      Element d(i,j) will be the Euclidean distance between two
%               column vectors X(:,i) and Y(:,j)
%
% Note:
%       The Euclidean distance D between two vectors X and Y is:
%       D = sum((x-y).^2).^0.5

[M, N] = size(x);
[M2, P] = size(y); 

if (M ~= M2)
    error('Matrix dimensions do not match.')
end

d = zeros(N, P);
if (N < P)
    copies = zeros(1,P);
    for n = 1:N
        d(n,:) = sum((x(:, n+copies) - y) .^2, 1);
    end
else
    copies = zeros(1,N);
    for p = 1:P
        d(:,p) = sum((x - y(:, p+copies)) .^2, 1)';
    end
end

d = d.^0.5;
end