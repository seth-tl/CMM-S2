
close all; clear all;
set(0,'defaulttextinterpreter','latex');

lls = linspace(1,1000, 1000);

figure()

experiment_name = '../../data/spectrum_experiment_multi_jet'
T = 10;

% figure()
tiledlayout(1,4,'TileSpacing','compact','Padding','none');
spectra = real(load(join([experiment_name, '.mat'])).spectra);


spectra0 = spectra(1,:);
spectra = spectra(2:end,:);
a = size(spectra); N = a(1)-1

N = 91
nexttile

for i = 1:N
  loglog(lls, spectra(i,:), 'Color', [0.219, 0.223, 0.698, 0.15 + (i+1)/2000]);  hold on;
  ylabel('Energy')
  xlabel('$\ell$')
  grid on
  axis tight
end

plot(lls(30:end), 200*lls(30:end).^(-3), '--k'), hold on
text(lls(500), 100*lls(500)^(-3), '$\ell^{-3}$', 'FontSize', 20), hold on

nexttile
% calculate means
spectrum = spectra(50,:);

for i = 51:N
  spectrum = spectrum + spectra(i,:);
end

mean_spectrum = spectrum./50;
% calculate standard deviation
std_spectrum = (spectra(50,:) - mean_spectrum).^2;

for i = 51:N
  std_spectrum = std_spectrum + (spectra(i,:)- mean_spectrum).^2;
end

std_spectrum = (std_spectrum./50).^(1/2); 

% standard_deviation
curve1 = mean_spectrum + std_spectrum;
curve2 = mean_spectrum - std_spectrum;
x2 = [lls, fliplr(lls)];
inBetween = [curve1, fliplr(curve2)];

loglog(lls, mean_spectrum, 'r-', 'LineWidth',1);  hold on;

fill(x2, inBetween, 'blue','FaceAlpha',0.3); hold on

%   set(gca, 'YScale', 'log')
ylabel('Energy')
xlabel('$\ell$')
grid on
axis tight

plot(lls(10:end), 100*lls(10:end).^(-3), '--k'), hold on
% plot(lls(5:75), 20000*lls(5:75).^(-5/3), '--k'), hold on
text(lls(500), 1000*lls(500)^(-3), '$\ell^{-3}$', 'FontSize', 20), hold on
% text(lls(20), 1000*lls(20)^(-3), '$\ell^{-5/3}$', 'FontSize', 20), hold on

% % xline(256,'--r')
% text(lls(30), lls(30)^(-3), '$t = 0.5 \to 1$', 'FontSize', 14)


energy = zeros(1,N);
for i = 1:N
  energy(1,i) = 0.5*sum(spectra(i,:));
end

nexttile
energy0 = real(0.5*sum(spectra0));

p = plot(linspace(0,T,N), (energy-energy0)/energy0, 'k');  hold on;


% ylim([-2e-3 2e-3])
ylabel('$(E(t)-E_0)/E_0$')
xlabel('time')

grid on


% legend('L = 512', 'L = 1024', 'L = 2048', 'L = 4096')


% figure()
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

plot(lls3(10:end), 10*lls3(10:end).^(-3), '--k')
text(lls2(500), 1000*lls2(500)^(-3), '$\ell^{-3}$', 'FontSize', 20)
% xline(256,'--r')


legend('L = 512', 'L = 1024', 'L = 2048', 'L = 4096', 'Location', 'southwest')

%set(gca, 'YScale', 'log')
ylabel('Energy')
xlabel('$\ell$')

grid on
axis tight


