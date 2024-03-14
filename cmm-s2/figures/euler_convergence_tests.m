clear all
close all
set(0,'defaulttextinterpreter','latex');

% Convergence Plots
% ==============================================================================
% RHwave unrotated and rotated
figure();

tlo = tiledlayout(2,4,'TileSpacing','compact','Padding','compact');

% RH wave tests
testRH = load('../data/errors/RHWave_tests_tscl_256_u512_T1.mat');
testRH_rotated = load('../data/errors/rhwave_rotated_tests_tscl_256_u512_T1.mat');
testRH_rotating = load('../data/errors/RHWave_rotating_tests_tscl_256_u512_T1.mat');

testRH_remap = load('../data/errors/RHWave_remapped_tests_tscl_256_u512_T1.mat');
testRH_rotated_remap = load('../data/errors/RHWave_rotated_remapped_tests_tscl_256_u512_T1.mat');
testRH_rotating_remap = load('../data/errors/RHWave_rotating_remapped_tests_tscl_256_u512_T1.mat');


%Reversing test
test_rev = load('../data/errors/gaussian_vortex_tests_tscl_256_u512_T1.mat');
test_rev_remap = load('../data/errors/gaussian_vortex_remapped_tests_tscl_256_u512_T1.mat');

% Steady Zonal Jet
% test_ZJ = load('../data/errors/ZJ_tests_tscl_256_u512_T1.mat');
% test_ZJ_rotating = load('../data/errors/ZJ_rotating_tests_tscl_256_u512_T0.1.mat');
% % test_ZJ_rotating_remap = load('../data/errors/ZJ_rotating_remapped_tests_tscl_256_u512_T1.mat');
% test_ZJ_rotating_remapped = load('../data/errors/ZJ_rotating_remapped_tests_tscl_256_u512_T0.1.mat');

test_ZJ_rotating = load('../data/zonaljet/errors_ZJ_tscl_256_u512_T0.5.mat');
test_ZJ_rotating_remapped = load('../data/zonaljet/errors_ZJ_remapping_tscl_256_u512_T0.5.mat');

Ns1 = testRH.edges;
Ns2 = test_ZJ_rotating.edges;

zwei = Ns1.^2;
drei = Ns1.^3;


nexttile
loglog(Ns1, testRH_rotating.linf, '--ko'), hold on;
loglog(Ns1, testRH_rotating.energy,'--kx'), hold on;
loglog(Ns1, testRH_rotating.enstrophy, '--k^'), hold on;
loglog(Ns1, 0.09*zwei, '-k');
grid on
legend({'vorticity','energy', 'enstrophy', 'h^2'}, 'Location','southeast', 'AutoUpdate','off');
loglog(Ns1, 90*zwei, '-k');

xlabel("$$ h $$");
ylabel("Error")
axis tight;
grid on
%
nexttile
loglog(Ns1, testRH_rotated.linf, '--ko'), hold on;
loglog(Ns1, testRH_rotated.energy,'--kx'), hold on;
loglog(Ns1, testRH_rotated.enstrophy, '--k^'), hold on;
loglog(Ns1, 0.7*zwei, '-k');
legend({'vorticity','energy', 'enstrophy', 'h^2'}, 'Location','southeast', 'AutoUpdate','off');

loglog(Ns1, 100*zwei, '-k');

grid on
axis tight
xlabel("$$ h $$");
ylabel("Error")

nexttile
% loglog(Ns1, test_rev.linf, '--ko'), hold on;
loglog(Ns1, test_rev.energy,'--kx'), hold on;
loglog(Ns1, test_rev.enstrophy, '--k^'), hold on;
% loglog(Ns1, test_rev.linf_map,'--ks'), hold on;
% loglog(Ns1, test_rev.linf_map(:,2), '--r+'), hold on;
% loglog(Ns1, test_rev.linf_map(:,3), '--rd'), hold on;

loglog(Ns1, 0.005*zwei, '-k');
% legend({'vorticity','energy', 'enstrophy', 'map-x', 'map-y', 'map-z', 'h^2'}, 'Location','southeast', 'AutoUpdate','off');
legend({'energy', 'enstrophy', 'h^2'}, 'Location','southeast', 'AutoUpdate','off');

loglog(Ns1, 0.4*zwei, '-k');

grid on
xlabel("$$ h $$");
ylabel("Error")
axis tight;



nexttile
loglog(Ns2, test_ZJ_rotating.linf, '--ko'), hold on;
loglog(Ns2, test_ZJ_rotating.energy,'--kx'), hold on;
loglog(Ns2, test_ZJ_rotating.enstrophy, '--k^'), hold on;
loglog(Ns1, 20*zwei, '-k');
loglog(Ns1, 0.05*zwei, '-k');

legend({'vorticity','energy', 'enstrophy', 'h^2'}, 'Location','southeast', 'AutoUpdate','off');

grid on
xlabel("$$ h $$");
ylabel("Error")
axis tight;
% then all the remapped tests

nexttile
loglog(Ns1, testRH_rotating_remap.linf, '--ko'), hold on;
loglog(Ns1, testRH_rotating_remap.energy,'--kx'), hold on;
loglog(Ns1, testRH_rotating_remap.enstrophy, '--k^'), hold on;
loglog(Ns1, 0.01*drei, '-k');
grid on
legend({'vorticity','energy', 'enstrophy', 'h^3'}, 'Location','southeast', 'AutoUpdate','off');
loglog(Ns1, 50*drei, '-k');

xlabel("$$ h $$");
ylabel("Error")
axis tight;
grid on
%
nexttile
loglog(testRH_rotated_remap.edges, testRH_rotated_remap.linf, '--ko'), hold on;
loglog(testRH_rotated_remap.edges, testRH_rotated_remap.energy,'--kx'), hold on;
loglog(testRH_rotated_remap.edges, testRH_rotated_remap.enstrophy, '--k^'), hold on;
loglog(Ns1, 0.5*drei, '-k');
legend({'vorticity','energy', 'enstrophy', 'h^3'}, 'Location','southeast', 'AutoUpdate','off');

loglog(Ns1, 50*drei, '-k');

grid on
axis tight
xlabel("$$ h $$");
ylabel("Error")

nexttile
% loglog(test_rev_remap.edges, test_rev_remap.linf, '--ko'), hold on;
loglog(test_rev_remap.edges, test_rev_remap.energy,'--kx'), hold on;
loglog(test_rev_remap.edges, test_rev_remap.enstrophy, '--k^'), hold on;
% loglog(test_rev_remap.edges, test_rev_remap.linf_map,'--ks'), hold on;


% loglog(Ns3, test_rev_remap.linf_map(:,2), '--r+'), hold on;
% loglog(Ns3, test_rev_remap.linf_map(:,3), '--rd'), hold on;

loglog(Ns1, 0.9*drei, '-k');
% legend({'vorticity','energy', 'enstrophy', 'map-x', 'map-y', 'map-z', 'h^2'}, 'Location','southeast', 'AutoUpdate','off');
legend({'energy', 'enstrophy', 'h^3'}, 'Location','southeast', 'AutoUpdate','off');

loglog(Ns1, 0.005*drei, '-k');

grid on
xlabel("$$ h $$");
ylabel("Error")
axis tight;



nexttile
loglog(Ns1, test_ZJ_rotating_remapped.linf, '--ko'), hold on;
loglog(Ns1, test_ZJ_rotating_remapped.energy,'--kx'), hold on;
loglog(Ns1, test_ZJ_rotating_remapped.enstrophy, '--k^'), hold on;
loglog(Ns1, 100*drei, '-k');
loglog(Ns1, 0.01*drei, '-k');

legend({'vorticity','energy', 'enstrophy', 'h^3'}, 'Location','southeast', 'AutoUpdate','off');

grid on
xlabel("$$ h $$");
ylabel("Error")
axis tight;



% % %
% % % % Zonal Jet Convergence --=================================================
% figure();
% tlo = tiledlayout(1,2,'TileSpacing','compact','Padding','compact');
% data1 = load('../EulerScripts/data/errors/errors_ZJ_tscl_256_u512_T0.1.mat')
% data2 = load('../EulerScripts/data/errors/errors_ZJ_remapping_tscl_256_u512_T0.1.mat')
%
%
% Ns = data1.edges
% zwei = Ns.^2;
%
% nexttile
% loglog(Ns, data1.linf, '--ko'), hold on;
% loglog(Ns, data1.energy,'--kx'), hold on;
% loglog(Ns, data1.enstrophy, '--k^'), hold on;
% loglog(Ns, 100*zwei, '-k');
% legend({'vorticity','energy', 'enstrophy', 'h^2'}, 'Location','northwest', 'AutoUpdate','off');
% % loglog(Ns, data2.linf, '--ro'), hold on;
% % loglog(Ns, data2.energy,'--rx'), hold on;
% % loglog(Ns, data2.enstrophy, '--r^'), hold on;
% grid on
% xlabel("$$ h $$");
% ylabel("Error")
% axis tight
%
% nexttile
% drei = Ns.^3;
%
% Ns2 = data2.edges;
% loglog(Ns, data2.linf, '--ko'), hold on;
% loglog(Ns2, data2.energy,'--kx'), hold on;
% loglog(Ns2, data2.enstrophy, '--k^'), hold on;
% loglog(Ns, 100*drei, '-k');
% legend({'vorticity', 'energy', 'enstrophy', 'h^3'}, 'Location','northwest', 'AutoUpdate','off');
%
% grid on
% xlabel("$$ h $$");
% ylabel("Error")
% axis tight
