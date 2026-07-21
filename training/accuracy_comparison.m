% accuracy_comparison.m — 训练前后准确率对比
words = categorical({'down','go','left','no','off','on','right','stop','up','yes'});
words = reordercats(words, {'down','go','left','no','off','on','right','stop','up','yes'});

before = [0, 100, 60, 0, 100, 0, 60, 0, 100, 100];
after  = [95.6, 94.8, 97.3, 97.8, 91.8, 93.7, 96.0, 99.0, 95.1, 98.3];

figure('Position', [100 100 900 500]);
b = bar([before' after'], 'grouped', 'BarWidth', 0.7);
b(1).FaceColor = [0.85 0.33 0.33];
b(2).FaceColor = [0.20 0.65 0.20];
set(gca, 'XTickLabel', {'down','go','left','no','off','on','right','stop','up','yes'});
ylabel('Accuracy (%)'); ylim([0 108]);
title('Per-Class Accuracy: Before vs After Fine-tuning', 'FontSize', 14);
legend({'Original (58%)', 'Fine-tuned (86.2%)'}, 'Location', 'southeast', 'FontSize', 11);
grid on;

% 标注数值
for i = 1:10
    text(i-0.18, before(i)+2, sprintf('%d%%', before(i)), 'FontSize', 8, 'Color', [0.8 0.2 0.2]);
    text(i+0.18, after(i)+2, sprintf('%.1f%%', after(i)), 'FontSize', 8, 'Color', [0.1 0.5 0.1]);
end

exportgraphics(gcf, 'accuracy_comparison.png', 'Resolution', 200);
disp('Saved: accuracy_comparison.png');
