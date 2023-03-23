clear all
close all

data = load('./data/density_correction_inexact_matching_N64.mat');
% data = load('./data/density_correction_advection.mat');

% data1 = load('./data/density_correction_k5.mat');
% data2= load('./data/density_correction_k6.mat');

figure()
tiledlayout(1,3,'TileSpacing','compact'); %,'Padding','none');

N = 1024;

Ji = reshape(data.dens_c, [N,N])';
Jc = reshape(data.rho_f, [N,N])';
Jf = reshape(data.dens_f, [N,N])';


x = linspace(-pi,pi, N);
y = linspace(-pi, pi, N);
[X,Y] = meshgrid(x,y);
cmax = max(Ji,[],'all');
cmin = min(Ji,[],'all');
%
%
% cmin = min([min(data.dens_i), min(data.dens_c), min(data.dens_f)])
% cmax = max([max(data.dens_i), max(data.dens_c), max(data.dens_f)])
nexttile
imagesc(x,y, Ji),colormap(magma), hold on;
% cb1 = colorbar('southoutside');
% cb1=colormap(jet(10));
% set(gca,'XTick',[0, pi/2, pi, 3*pi/2, 2*pi]);
% set(gca,'YTick',[-pi, pi]);
% xticklabels({'0','\pi /2', '\pi', '3\pi/2', '2\pi'});
% yticklabels({'0', '\pi'});
% set(gca, 'CLim', [cmin, cmax])

% axis square;\
colorbar('southoutside')

nexttile
imagesc(x,y, Jc),colormap(magma), hold on;
% cb1 = colorbar('southoutside');
% cb1=colormap(jet(10));
% set(gca,'XTick',[0, pi/2, pi, 3*pi/2, 2*pi]);
% set(gca,'YTick',[0, pi]);
% xticklabels({'0','\pi /2', '\pi', '3\pi/2', '2\pi'});
% yticklabels({'0', '\pi'});
% % axis square;
colorbar('southoutside')
% set(gca, 'CLim', [cmin, cmax])


nexttile
imagesc(x,y, Jf); colormap(magma), hold on;
% cb1 = colorbar('southoutside');
% % cb1=colormap(jet(10));
% set(gca,'XTick',[0, pi/2, pi, 3*pi/2, 2*pi]);
% set(gca,'YTick',[0, pi]);
% xticklabels({'0','\pi /2', '\pi', '3\pi/2', '2\pi'});
% yticklabels({'0', '\pi'});
% set(gca, 'CLim', [cmin, cmax + 1000])

% axis square;
colorbar('southoutside')

%
% nexttile
% imagesc(x,y, reshape(data.q1, [500,500])'),colormap(jet(1024)), hold on;
% % cb1 = colorbar('southoutside');
% % cb1=colormap(jet(10));
% set(gca,'XTick',[0, pi/2, pi, 3*pi/2, 2*pi]);
% set(gca,'YTick',[0, pi]);
% xticklabels({'0','\pi /2', '\pi', '3\pi/2', '2\pi'});
% yticklabels({'0', '\pi'});
% % axis square;
% nexttile
% imagesc(x,y, reshape(data.q_i, [500,500])'),colormap(jet(1024)), hold on;
% % cb1 = colorbar('southoutside');
% % cb1=colormap(jet(10));
% set(gca,'XTick',[0, pi/2, pi, 3*pi/2, 2*pi]);
% set(gca,'YTick',[0, pi]);
% xticklabels({'0','\pi /2', '\pi', '3\pi/2', '2\pi'});
% yticklabels({'0', '\pi'});
% % axis square;
% nexttile
% imagesc(x,y, reshape(data.q_f, [500,500])'),colormap(jet(1024)), hold on;
% % cb1 = colorbar('southoutside');
% % cb1=colormap(jet(10));
% set(gca,'XTick',[0, pi/2, pi, 3*pi/2, 2*pi]);
% set(gca,'YTick',[0, pi]);
% xticklabels({'0','\pi /2', '\pi', '3\pi/2', '2\pi'});
% yticklabels({'0', '\pi'});
