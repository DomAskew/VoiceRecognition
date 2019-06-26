clear all;clc;
close all

% https://stackoverflow.com/questions/24195089/remove-noise-from-wav-file-matlab#


[f, fs] = audioread('1s4.wav');

soundsc(f,fs);

% Plot both audio channels
N = size(f,1); 
figure;
subplot(2,1,1);
stem(1:N, f(:,1));
title('Left Channel');
subplot(2,1,2);
stem(1:N, f(:,2));
title('Right Channel');


pause(1)

n = 7;



beginFreq = 7000 / (fs/2);
endFreq = 10000 / (fs/2);
% [b,a] = butter(n, [beginFreq, endFreq], 'bandpass'); % if a range is
% required

[b,a] = butter(n,beginFreq)

fOut = filter(b, a, f);

soundsc(fOut,fs)