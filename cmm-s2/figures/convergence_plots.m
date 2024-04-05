clear all
close all
set(0,'defaulttextinterpreter','latex');

% Convergence Plots
% ==============================================================================
figure();

tlo = tiledlayout(2,4,'TileSpacing','compact','Padding','compact');

% RH wave tests
data_rh  = load('../../data/errors_convergence_test_rossby_wave.mat');
data_rh_static = load('../../data/errors_convergence_test_rossby_wave_static.mat');
data_gv = load('../../data/errors_convergence_test_gaussian_vortex.mat');
data_zj = load('../../data/errors_convergence_test_zonal_jet.mat');

Ns1 = data_rh.edges;

zwei = Ns1.^2;
drei = Ns1.^3;

nexttile
loglog(Ns1, data_rh.linf, '--ko'), hold on;
loglog(Ns1, data_rh.energy,'--rx'), hold on;
loglog(Ns1, data_rh.enstrophy, '--b^'), hold on;
loglog(Ns1, 0.09*zwei, '-k');
grid on
legend({'vorticity','energy', 'enstrophy', 'h^2'}, 'Location','northwest','FontSize',16, 'AutoUpdate','off');
loglog(Ns1, 90*zwei, '-k');

xlabel("$$ h $$");
ylabel("Error")
axis tight;
grid on
%
nexttile
loglog(Ns1, data_rh_static.linf, '--ko'), hold on;
loglog(Ns1, data_rh_static.energy,'--rx'), hold on;
loglog(Ns1, data_rh_static.enstrophy, '--b^'), hold on;
loglog(Ns1, 0.7*zwei, '-k');
legend({'vorticity','energy', 'enstrophy', 'h^2'}, 'Location','northwest','FontSize',16, 'AutoUpdate','off');

loglog(Ns1, 100*zwei, '-k');

grid on
axis tight
xlabel("$$ h $$");
ylabel("Error")

nexttile
loglog(Ns1, data_gv.energy,'--rx'), hold on;
loglog(Ns1, data_gv.enstrophy, '--b^'), hold on;

loglog(Ns1, 0.005*zwei, '-k');
legend({'energy', 'enstrophy', 'h^2'}, 'Location','northwest','FontSize',16, 'AutoUpdate','off');
loglog(Ns1, 0.4*zwei, '-k');

grid on
xlabel("$$ h $$");
ylabel("Error")
axis tight;

nexttile
loglog(Ns1, data_zj.linf, '--ko'), hold on;
loglog(Ns1, data_zj.energy,'--rx'), hold on;
loglog(Ns1, data_zj.enstrophy, '--b^'), hold on;
loglog(Ns1, 300*zwei, '-k');
loglog(Ns1, 0.1*zwei, '-k');

legend({'vorticity','energy', 'enstrophy', 'h^2'}, 'Location','northwest','FontSize',16, 'AutoUpdate','off');

grid on
xlabel("$$ h $$");
ylabel("Error")
axis tight;


% then all the remapped tests:

% RH wave tests
data_rh  = load('../../data/errors_convergence_test_rossby_wave_remapping.mat');
data_rh_static = load('../../data/errors_convergence_test_rossby_wave_static_remapping.mat');
data_gv = load('../../data/errors_convergence_test_gaussian_vortex_remapping.mat');
data_zj = load('../../data/errors_convergence_test_zonal_jet_remapping.mat');


nexttile
loglog(Ns1, data_rh.linf, '--ko'), hold on;
loglog(Ns1, data_rh.energy,'--rx'), hold on;
loglog(Ns1, data_rh.enstrophy, '--b^'), hold on;
loglog(Ns1, 0.01*drei, '-k');
grid on
legend({'vorticity','energy', 'enstrophy', 'h^3'}, 'Location','northwest','FontSize',16, 'AutoUpdate','off');
loglog(Ns1, 50*drei, '-k');

xlabel("$$ h $$");
ylabel("Error")
axis tight;
grid on
%
nexttile
loglog(Ns1, data_rh_static.linf, '--ko'), hold on;
loglog(Ns1, data_rh_static.energy,'--rx'), hold on;
loglog(Ns1, data_rh_static.enstrophy, '--b^'), hold on;
loglog(Ns1, 0.5*drei, '-k');
legend({'vorticity','energy', 'enstrophy', 'h^3'}, 'Location','northwest','FontSize',16, 'AutoUpdate','off');

loglog(Ns1, 50*drei, '-k');

grid on
axis tight
xlabel("$$ h $$");
ylabel("Error")

nexttile
loglog(Ns1, data_gv.energy,'--rx'), hold on;
loglog(Ns1, data_gv.enstrophy, '--b^'), hold on;
loglog(Ns1, 0.9*drei, '-k');
legend({'energy', 'enstrophy', 'h^3'}, 'Location','northwest','FontSize',16, 'AutoUpdate','off');

loglog(Ns1, 0.005*drei, '-k');

grid on
xlabel("$$ h $$");
ylabel("Error")
axis tight;
%
nexttile
loglog(Ns1, data_zj.linf, '--ko'), hold on;
loglog(Ns1, data_zj.energy,'--rx'), hold on;
loglog(Ns1, data_zj.enstrophy, '--b^'), hold on;
loglog(Ns1, 2000*drei, '-k');
loglog(Ns1, 0.5*drei, '-k');

legend({'vorticity','energy', 'enstrophy', 'h^3'}, 'Location','northwest','FontSize',16, 'AutoUpdate','off');

grid on
xlabel("$$ h $$");
ylabel("Error")
axis tight;
