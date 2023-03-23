close all
set(0,'defaulttextinterpreter','latex');

data = load('./data/density_correction_errors_generic.mat');
data2 = load('../SDiff_Projection/data/density_correction_errors_generic_L512.mat');

tiledlayout(1,2,'TileSpacing','compact', 'Padding','compact')

errors = data.errors;

Ns = [16, 32, 64, 128, 256]./(2*pi);
nexttile
loglog(Ns.^(-1), errors(1:5), 'ko--'); hold on
loglog(Ns.^(-1), 0.03*Ns.^(-3), 'k-')
grid on
legend('Error', 'h^3', 'Location','southeast')
xlabel('h')
ylabel('Correction Error')

nexttile
hs = data2.hs;

loglog(hs, data2.errors, 'ko--'); hold on
loglog(hs(1:4), 0.2*hs(1:4).^(2), 'k-')
loglog(hs(4:end), 0.03*hs(4:end).^(1), 'k--')
legend('Error', 'h^2', 'h^1', 'Location','southeast')


axis tight
grid on
xlabel('h')
ylabel('Correction Error')


figure()
data = load('../SDiff_Projection/data/density_correction_errors_L1000.mat');
data2 = load('../SDiff_Projection/data/density_correction_errors.mat');
data3 = load('../SDiff_Projection/data/density_correction_SDiff_errors_L256.mat')
data4 = load('../SDiff_Projection/data/density_correction_SDiff_errors_L256.mat')

%
% tiledlayout(1,2,'TileSpacing','compact', 'Padding','compact')

error1 = data.errors;
error1_l2 = data.mass_error;


error2 = data2.errors;
error2_l2 = data2.mass_error;

hs = data2.hs;
nexttile

loglog(hs, error1, 'kx--'); hold on
loglog(hs, error2, 'ko--'); hold on
loglog(hs, error1_l2, 'bx--'); hold on
loglog(hs, error2_l2, 'bo--'); hold on
loglog(hs, data3.mass_error, 'rx--'); hold on
loglog(hs, data4.mass_error, 'ro--'); hold on

loglog(hs, 3*hs, 'k--')
loglog(hs, 0.03*hs.^(2), 'k-')


grid on
legend('l^{\infty} (L = 256)', 'l^{\infty} (L = 512)', 'mean (L = 256)', 'mean (L = 512)', 'SDiff (L = 256)', 'SDiff (L = 512)','h^1','h^2', 'Location','southeast')
xlabel('h')
ylabel('Correction Error')
axis tight
