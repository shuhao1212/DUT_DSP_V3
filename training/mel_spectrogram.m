% mel_spectrogram.m — Mel 频谱可视化 (生成样本数据)
% 用正弦扫频模拟真实 Mel 谱外观

mel = zeros(40, 101);
for f = 1:40
    freq = 20 + f * 5;
    for t = 1:101
        mel(f, t) = cos(2*pi*freq*t/101) * exp(-0.5*((t-50)/25)^2) * (1 - f/50);
    end
end
mel = mel + randn(40, 101) * 0.3;
mel = mel - min(mel(:));
mel = mel / max(mel(:));

figure('Position', [100 100 800 300]);
imagesc(mel);
colormap(hot);
c = colorbar;
c.Label.String = 'Log Energy (dB)';
xlabel('Frame (101 frames = 1 sec)');
ylabel('Mel Filter Bank (0-40)');
title('Log-Mel Spectrogram Example', 'FontSize', 14);

exportgraphics(gcf, 'mel_spectrogram.png', 'Resolution', 200);
disp('Saved: mel_spectrogram.png');
