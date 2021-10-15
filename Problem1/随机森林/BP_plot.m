sample = 1:1:395;
plot(sample,testtarget,'red');
hold on
plot(sample,Yq_test,'blue');
legend('Target','Predict','Fontname', 'Times New Roman','FontSize',14);

figure

sample = 1:1:1579;
plot(sample,Y,'red');
hold on
plot(sample,Yq,'blue');
legend('Target','Predict','Fontname', 'Times New Roman','FontSize',14);