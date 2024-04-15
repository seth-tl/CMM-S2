close all; clear all


t_init = 2; t_inter = 5; t_final = 10;
set(0,'defaulttextinterpreter','latex');

experiment_name = '../../data/spectrum_experiment_condensate_rotating'
spectra = real(load(join([experiment_name, '.mat'])).spectra);
samples = load('../../data/condensate_rotating_experiment_voriticity_figures.mat').omgs;

spectra0 = spectra(1,:);
spectra = spectra(2:end,:);
a = size(spectra); N = a(1)-1;

figure = tiledlayout(2,3,'TileSpacing','compact','Padding','none')

% t_init data


nexttile 
imagesc(squeeze(samples(t_init,:,:))); hold on
shading interp
colormap(inferno);
%     spherefun.plotEarth('k-');
axis square
axis tight
xlabel('$\lambda$')
ylabel('$\theta$')
set(gca,'xtick',[])
set(gca,'xticklabel',[])
set(gca,'ytick',[])
set(gca,'yticklabel',[])
% view([lon,lat])
title(join(['t = ' , num2str(t_init*10)]),"FontSize",20)

nexttile
% p1 = surf(x_s,y_s,z_s, omega); hold on
imagesc(squeeze(samples(t_inter,:,:))); hold on
shading interp
colormap(inferno);
%     spherefun.plotEarth('k-');
axis square
axis tight
xlabel('$\lambda$')
ylabel('$\theta$')
set(gca,'xtick',[])
set(gca,'xticklabel',[])
set(gca,'ytick',[])
set(gca,'yticklabel',[])
title(join(['t = ' , num2str(t_inter*10)]),"FontSize", 20)

nexttile
% p1 = surf(x_s,y_s,z_s, omega); hold on
imagesc(squeeze(samples(t_final,:,:))); hold on
shading interp
colormap(inferno);
%     spherefun.plotEarth('k-');
axis square
axis tight
xlabel('$\lambda$')
ylabel('$\theta$')
set(gca,'xtick',[])
set(gca,'xticklabel',[])
set(gca,'ytick',[])
set(gca,'yticklabel',[])
title(join(['t = ' , num2str(100)]),"FontSize", 20)


nexttile
lls = linspace(1,1000, 1000);
for i = 1:30
  p = loglog(lls, spectra(i,:), 'Color', [0.219, 0.223, 0.698, 0.15 + (i+1)/100]);  hold on;
  ylabel('Energy', 'FontSize', 20)
  xlabel('$\ell$', 'FontSize', 20)
  grid on
  axis tight
  j = j+1
end

plot(lls, 200*lls.^(-3), '--k'), hold on
text(lls(500), 1000*lls(500)^(-3), '$\ell^{-3}$', 'FontSize', 20), hold on

nexttile
for i = 30:N
  p = loglog(lls, spectra(i,:), 'Color', [0.219, 0.223, 0.698, 0.15 + (i+1)/1000]);  hold on;
%   set(gca, 'YScale', 'log')
  ylabel('Energy', 'FontSize', 20)
  xlabel('$\ell$', 'FontSize', 20)
  grid on
  axis tight
end

plot(lls, 2000*lls.^(-5), '--k'), hold on
plot(lls, 0.0001*lls.^(-1), '--r'), hold on
text(lls(500), 1000*lls(500)^(-5), '$\ell^{-5}$', 'FontSize', 20), hold on
text(lls(500), 1000*lls(500)^(-1), '$\ell^{-1}$', 'FontSize', 20), hold on


nexttile

file0 = join([experiment_name,'_upsampling512.mat']);
file1 = join([experiment_name,'_upsampling1024.mat']);
file2 = join([experiment_name,'_upsampling2048.mat']);
file3 = join([experiment_name,'_upsampling4096.mat']);

spectrum0 = load(file0).spectra;
spectrum1 = load(file1).spectra;
spectrum2 = load(file2).spectra;
spectrum3 = load(file3).spectra;

lls0 = linspace(1,512,512);
lls1 = linspace(1,1024,1024);
lls2 = linspace(1,2048,2048);
lls3 = linspace(1,4096,4096);

loglog(lls0, spectrum0(1,:), 'Color', [0.68,0.08,0.09]);  hold on;
loglog(lls1, spectrum1(1,:),  'Color', [0.11,0.7,0.29]);  hold on;
loglog(lls2, spectrum2(1,:),  'Color', [0.22,0.27,0.74]);  hold on;
loglog(lls3, spectrum3(1,:),  'Color', [0.46,0.11,0.39]);  hold on;

plot(lls3, 0.000008*lls3.^(-1), '--r'), hold on
text(lls(500), 0.01*lls(500)^(-1), '$\ell^{-1}$', 'FontSize', 20), hold on


legend({'L = 512', 'L = 1024', 'L = 2048', 'L = 4096'}, 'FontSize', 16, 'Location', 'southwest')

ylabel('Energy', 'FontSize', 20)
xlabel('$\ell$', 'FontSize', 20)

grid on
axis tight


