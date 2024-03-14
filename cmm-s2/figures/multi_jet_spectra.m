
close all; clear all;
set(0,'defaulttextinterpreter','latex');

lls = linspace(1,1024, 1024);
j_list = 10:10:990
N = length(j_list)
figure()

% A 3D wavetable type plot
% concatenate an array
% data_array = zeros(N,2000);
%
% for i = 1:N
%   k = j_list(i)
%   file = join(['../EulerScripts/data/simulations/spectra/zonaljet_sim2_spectra/spectrum_omega_ZJsim2_unperturbed_zonal_jet_ures200_', num2str(k) , '.mat']);
%   spectrum = load(file).ells;
%   data_array(i,:) = spectrum;
% end
%
% surf(j_list, lls, data_array');
%
% xlabel('time')
% ylabel('$\ell$')
% set(gca, 'ZScale', 'log')
% m = min(data_array)
% clim([m(end), 10^-3])
% set(gca,'ColorScale','log')
% shading flat
% colormap magma

% figure()
tiledlayout(1,4,'TileSpacing','compact','Padding','none');

% j_list2 = [70, 140, 210, 260, 330, 400, 430, 460, 490, 520, 550, 580, 610, 630];
% j_list2 = 50:50:690
nexttile
j_list = 10:10:990
N2 = length(j_list)
for i = 1:N2
  k = j_list(i);
  file = join(['../data/simulations/spectra/multi_jet_spectra/spectrum_omega_rotating_remapped_multi_jet_tscl1000_ures256_T1_', num2str(k) , '.mat']);
  spectrum = load(file).ells;
  p = loglog(lls, spectrum(1,:), 'Color', [0.219, 0.223, 0.698, 0.15 + (i+1)/1000]);  hold on;
%   set(gca, 'YScale', 'log')
  ylabel('Energy')
  xlabel('$\ell$')
  grid on
  axis tight
end

plot(lls(30:end), 20000*lls(30:end).^(-3), '--k'), hold on
text(lls(500), 1000*lls(500)^(-3), '$\ell^{-3}$', 'FontSize', 20), hold on
% plot(lls(3:60), 700*lls(3:60).^(-5/3), '--k'), hold on
% text(lls(20), 1000*lls(20)^(-3), '$\ell^{-5/3}$', 'FontSize', 20), hold on

% text(lls(30), lls(30)^(-3), '$t = 0 \to 0.5$', 'FontSize', 14)


nexttile
j_list = 500:10:990
N2 = length(j_list)
% add together and show spectrum plot with errors
k = j_list(1);
file = join(['../data/simulations/spectra/multi_jet_spectra/spectrum_omega_rotating_remapped_multi_jet_tscl1000_ures256_T1_', num2str(k) , '.mat']);
spectrum = load(file).ells;
% calculate means
for i = 2:N2
  k = j_list(i);
  file = join(['../data/simulations/spectra/multi_jet_spectra/spectrum_omega_rotating_remapped_multi_jet_tscl1000_ures256_T1_', num2str(k) , '.mat']);
  spectrum = spectrum + load(file).ells;
end

mean_spectrum = spectrum./N2
% calculate standard deviation
k = j_list(1);
file = join(['../data/simulations/spectra/multi_jet_spectra/spectrum_omega_rotating_remapped_multi_jet_tscl1000_ures256_T1_', num2str(k) , '.mat']);
std_spectrum = (load(file).ells - mean_spectrum).^2;

for i = 2:N2
  k = j_list(i);
  file = join(['../data/simulations/spectra/multi_jet_spectra/spectrum_omega_rotating_remapped_multi_jet_tscl1000_ures256_T1_', num2str(k) , '.mat']);
  std_spectrum = std_spectrum + (load(file).ells - mean_spectrum).^2;
end

std_spectrum = (std_spectrum./N2).^(1/2)

% standard_deviation
curve1 = mean_spectrum + std_spectrum;
curve2 = mean_spectrum - std_spectrum;
x2 = [lls, fliplr(lls)];
inBetween = [curve1, fliplr(curve2)];


p = loglog(lls, mean_spectrum, 'r-', 'LineWidth',1);  hold on;

fill(x2, inBetween, 'blue','FaceAlpha',0.3); hold on

%   set(gca, 'YScale', 'log')
ylabel('Energy')
xlabel('$\ell$')
grid on
axis tight

plot(lls(10:end), 20000*lls(10:end).^(-3), '--k'), hold on
% plot(lls(5:75), 20000*lls(5:75).^(-5/3), '--k'), hold on
text(lls(500), 1000*lls(500)^(-3), '$\ell^{-3}$', 'FontSize', 20), hold on
% text(lls(20), 1000*lls(20)^(-3), '$\ell^{-5/3}$', 'FontSize', 20), hold on

% % xline(256,'--r')
% text(lls(30), lls(30)^(-3), '$t = 0.5 \to 1$', 'FontSize', 14)


energy = zeros(1,N)
j_list = 10:10:990
for i = 1:N
  k = j_list(i);
  file = join(['../data/simulations/spectra/multi_jet_spectra/spectrum_omega_rotating_remapped_multi_jet_tscl1000_ures256_T1_', num2str(k) , '.mat']);
  spectrum = load(file).ells;
  energy(1,i) = 0.5*sum(spectrum(1,:));
end

nexttile
T = 1
ymax = min((energy-energy(1))/energy(1))
p = plot(linspace(0,T,N), (energy-energy(1))/energy(1), 'k');  hold on;
plot(linspace(0.2,T,N), (0.19*linspace(0.2,T,N)).^3, 'k--');

% set(gca, 'Ylimit', [ymax, -ymax])
% text(0.5, (0.22*0.5)^(3), '$\ell^{-3}$', 'FontSize', 10)
ylim([1e-9 1e-3])
ylabel('$(E(t)-E_0)/E_0$')
xlabel('time')

grid on
axis tight


% legend('L = 512', 'L = 1024', 'L = 2048', 'L = 4096')


% figure()
nexttile
file0 = '../data/simulations/spectra/multi_jet_spectra/subgrid_spectrum_omega_rotating_remapped_multi_jet_ures256_T1_512.mat';
file1 = '../data/simulations/spectra/multi_jet_spectra/subgrid_spectrum_omega_rotating_remapped_multi_jet_ures256_T1_1024.mat';
file2 = '../data/simulations/spectra/multi_jet_spectra/subgrid_spectrum_omega_rotating_remapped_multi_jet_ures256_T1_2048.mat';
file3 = '../data/simulations/spectra/multi_jet_spectra/subgrid_spectrum_omega_rotating_remapped_multi_jet_ures256_T1_4096.mat';

spectrum0 = load(file0).ells;
spectrum1 = load(file1).ells;
spectrum2 = load(file2).ells;
spectrum3 = load(file3).ells;

lls0 = linspace(1,512,512);
lls1 = linspace(1,1024,1024);
lls2 = linspace(1,2048,2048);
lls3 = linspace(1,4096,4096);

loglog(lls0, spectrum0(1,:), 'Color', [0.68,0.08,0.09]);  hold on;
loglog(lls1, spectrum1(1,:),  'Color', [0.11,0.7,0.29]);  hold on;
loglog(lls2, spectrum2(1,:),  'Color', [0.22,0.27,0.74]);  hold on;
loglog(lls3, spectrum3(1,:),  'Color', [0.46,0.11,0.39]);  hold on;

plot(lls2(10:end), 700*lls2(10:end).^(-3), '--k')
text(lls2(500), 1000*lls2(500)^(-3), '$\ell^{-3}$', 'FontSize', 20)
% xline(256,'--r')


legend('L = 512', 'L = 1024', 'L = 2048', 'L = 4096', 'Location', 'southwest')

%set(gca, 'YScale', 'log')
ylabel('Energy')
xlabel('$\ell$')

grid on
axis tight



% decay = 10000000*lls.^(-3)
% loglog(lls(1,100:end), decay(1,100:end),'k')

% legend({'N_t=70', 'N_t=140', 'N_t=210', 'N_t=260', 'N_t=330', 'N_t=400', 'N_t=470', 'N_t=540', 'N_t=635'}, 'Location', 'southwest')
%
% nexttile
% for i = 1:9
%   k = j_list(i)
%   file = join(['../MainScripts/data/simulations/spectra/spectrum_omega_', 'steady_zonal_jet', '_shifted_ures256_', num2str(k) , '.mat']);
%   spectrum = load(file).ells;
%   p = loglog(lls, sqrt(spectrum(1,:)), 'Color', [0.1, 0, 0.5, (i+1)/10]);  hold on;
%   %set(gca, 'YScale', 'log')
%   ylabel('Power')
%   xlabel('$\ell$')
%
% end
% % decay = 10000000*lls.^(-3)
% % loglog(lls(1,100:end), decay(1,100:end),'k')
%
% legend({'N_t=70', 'N_t=140', 'N_t=210', 'N_t=260', 'N_t=330', 'N_t=400', 'N_t=470', 'N_t=540', 'N_t=635'}, 'Location', 'southwest')
