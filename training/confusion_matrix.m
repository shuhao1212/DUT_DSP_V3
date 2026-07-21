% confusion_matrix.m — 12 类混淆矩阵热力图
labels = {'silence','unknown','down','go','left','no','off','on','right','stop','up','yes'};

cm = [
    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0;
    0, 5573,  208,  193,  118,  191,   46,   96,  304,  110,   43,   49;
    0,    1,  388,    8,    0,    8,    0,    0,    0,    0,    0,    1;
    0,    4,    8,  381,    0,    8,    1,    0,    0,    0,    0,    0;
    0,    3,    0,    1,  401,    2,    0,    0,    1,    0,    0,    4;
    0,    0,    2,    3,    2,  396,    0,    0,    0,    0,    1,    1;
    0,    2,    0,    8,    1,    1,  369,    5,    0,    1,   15,    0;
    0,    7,    5,    1,    0,    0,    7,  371,    0,    1,    4,    0;
    0,    9,    0,    3,    2,    0,    1,    0,  380,    1,    0,    0;
    0,    3,    0,    1,    0,    0,    0,    0,    0,  407,    0,    0;
    0,    4,    2,    2,    0,    0,    3,    7,    0,    3,  404,    0;
    0,    4,    0,    0,    1,    0,    0,    0,    2,    0,    0,  412
];

cm_pct = cm ./ sum(cm, 2) * 100;

figure('Position', [100 100 900 750]);
h = heatmap(labels, labels, cm_pct, 'Colormap', parula, 'CellLabelColor','none');
h.Title = 'Confusion Matrix — BC-ResNet KWS (Test Acc: 86.2%)';
h.XLabel = 'Predicted';
h.YLabel = 'True Label';

exportgraphics(gcf, 'confusion_matrix.png', 'Resolution', 200);
disp('Saved: confusion_matrix.png');
