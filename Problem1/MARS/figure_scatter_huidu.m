figure(1);
x = [0:0.01:3];
y = [0:0.01:3];
scatter(Yt,Yq_test,'black');
hold on 
% annotation('R^2=0.6696',[0.2,0.2,0.3,0.4],'LineStyle','-','LineWidth',2)
plot(x*5,y*5,'-r','LineWidth',2);

axis([0 12 0 20])
% plot(t,Z1(:,26),'-r','LineWidth',2);
txt = 'R^2=0.9028';
text(150,280,txt,'fontsize',14,'fontname','Times New Roman')
hold on
% plot(t,Z2(:,26),'-b','LineWidth',2);
set(gca,'linewidth',1,'fontsize',14,'fontname','Times New Roman');
% legend('','R^2=0.6696','Fontname', 'Times New Roman','FontSize',14);
title(['Comparison of measured and MARS-predicted {BPF}'],'Fontname', 'Times New Roman','FontSize',14);
xlabel('Target','Fontname', 'Times New Roman','FontSize',14); % xlabel('x','Fontname', 'Times New Roman','FontSize',12);
ylabel('Prediction m','Fontname', 'Times New Roman','FontSize',14);
grid on