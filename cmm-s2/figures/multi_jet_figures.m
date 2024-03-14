close all
set(0,'defaulttextinterpreter','latex');

load spine
% v = VideoWriter('../EulerScripts/data/videos/remapped_perturbed_zonal_jet_T1.avi', 'Motion JPEG AVI');
% open(v);

lat = 55;
lon = 10;
cam_rot_rate = 0.1

N_pts = 1000
phi_grid = linspace(0, 2*pi, N_pts);
theta_grid = linspace(0, pi, N_pts);
[phi, theta] = meshgrid(phi_grid, theta_grid);
%[x_s,y_s,z_s]=sphere(npixels);
x_s = sin(theta).*cos(phi);
y_s = sin(theta).*sin(phi);
z_s = cos(theta);


% j = 0
% figure()
% tiledlayout(2,3, 'TileSpacing','compact','Padding','none');
%
% % ts = ['t=2T/7', 't=3T/7', 't=4T/7', 't=5T/7', 't=6T/7'];
% i = 2
% for k = 200:100:600
%
%      % data_tr = load(join(['../data/simulations/multi_jet/figures/passive_tracer_remapped_multi_jet_ures256_T1_', num2str(k) , '.mat']));
%      data_omega = load(join(['../data/simulations/multi_jet/advected_quantities/omega_rotating_remapped_multi_jet_ures256_T1_', num2str(k) , '.mat']));
%      % data_omegaW = load(join(['../EulerScripts/data/simulations/multi_jet/advected_quantities/window_omega_remapped_multi_jet_ures256_T1_', num2str(k) , '.mat']));
%
%      % tracer_g = data_tr.glob; %tracer_w1 = data_tr.window1; tracer_w2 = data_tr.window2;
%      omega_g = data_omega.glob;
%      % omega_w1 = data_omegaW.window1; omega_w2 = data_omegaW.window2;
%
%      nexttile
%      imagesc(phi_grid, theta_grid, omega_g)
%      colormap(jet(1024))
%      xlabel('$\lambda$')
%      ylabel('$\theta$')
%      axis square
%      axis tight
%      title(join(['t=',num2str(i),'T/7']))
%      i = i+1
% end

k = 695
% data_tr = load(join(['../data/simulations/multi_jet/figures/passive_tracer_remapped_multi_jet_ures256_T1_', num2str(k) , '.mat']));
data_omega = load(join(['../data/simulations/multi_jet/advected_quantities/omega_rotating_remapped_multi_jet_ures256_T1_', num2str(k) , '.mat']));
 % data_omegaW = load(join(['../EulerScripts/data/simulations/multi_jet/advected_quantities/window_omega_remapped_multi_jet_ures256_T1_', num2str(k) , '.mat']));

 j = 0
 figure()
 tiledlayout(2,3, 'TileSpacing','compact','Padding','none');


% tracer_g = data_tr.glob; %tracer_w1 = data_tr.window1; tracer_w2 = data_tr.window2;
omega_g = data_omega.glob;

nexttile
imagesc(phi_grid,theta_grid,omega_g), hold on
colormap(jet(1024))
%
% omega_w1 = data_omegaW.window1; omega_w2 = data_omegaW.window2;
phi0 = 3.1699; theta0 =  0.478;
phi_b = linspace(phi0 - 1/4, phi0 + 1/4);
theta_l = linspace(theta0 - 1/4, theta0 + 1/4);

uns = ones(length(phi_b));
plot(uns*(phi0-1/4), theta_l,'w-', 'LineWidth',1.2), hold on
plot(uns*(phi0+1/4), theta_l,'w-', 'LineWidth',1.2), hold on
plot(phi_b, (theta0+1/4)*uns,'w-', 'LineWidth',1.2), hold on
plot(phi_b, (theta0-1/4)*uns,'w-', 'LineWidth',1.2), hold on



xlabel('$\lambda$')
ylabel('$\theta$')
axis square
axis tight
set(gca,'XTick',[0, pi/2, pi, 3*pi/2, 2*pi]);
set(gca,'YTick',[0, pi]);
xticklabels({'0', '\pi/2', '\pi', '3\pi/2', '2\pi'});
yticklabels({'0', '\pi'});

title('t=T')
%

% Zoom windows

phis1 = linspace(phi0- 2^-3, phi0 + 2^-3,N_pts);
thetas1 = linspace(theta0 - 2^-3, theta0 + 2^-3, N_pts);
[X1,Y1] = meshgrid(phis1, thetas1);

%%window2

phis2 = linspace(phi0- 2^-5, phi0 + 2^-5,N_pts);
thetas2 = linspace(theta0- 2^-5, theta0 + 2^-5, N_pts);
[X2,Y2] = meshgrid(phis2, thetas2);


phis3 = linspace(phi0- 2^-7, phi0 + 2^-7,N_pts);
thetas3 = linspace(theta0- 2^-7,theta0 + 2^-7, N_pts);
[X3,Y3] = meshgrid(phis3, thetas3);

phis4 = linspace(phi0- 2^-9, phi0 + 2^-9,N_pts);
thetas4 = linspace(theta0- 2^-9,theta0 + 2^-9, N_pts);
[X4,Y4] = meshgrid(phis4, thetas4);


phis5 = linspace(phi0- 2^-11, phi0 + 2^-11,N_pts);
thetas5 = linspace(theta0- 2^-11,theta0 + 2^-11, N_pts);
[X5,Y5] = meshgrid(phis5, thetas5);



data_omegaw1 = load(join(['../data/simulations/multi_jet/figures/window1_omega_rotating_remapped_multi_jet_ures256_T1_', num2str(k) , '.mat']));
data_omegaw2 = load(join(['../data/simulations/multi_jet/figures/window2_omega_rotating_remapped_multi_jet_ures256_T1_', num2str(k) , '.mat']));
data_omegaw3 = load(join(['../data/simulations/multi_jet/figures/window3_omega_rotating_remapped_multi_jet_ures256_T1_', num2str(k) , '.mat']));
data_omegaw4 = load(join(['../data/simulations/multi_jet/figures/window4_omega_rotating_remapped_multi_jet_ures256_T1_', num2str(k) , '.mat']));
data_omegaw5 = load(join(['../data/simulations/multi_jet/figures/window5_omega_rotating_remapped_multi_jet_ures256_T1_', num2str(k) , '.mat']));
% data_omegaw6 = load(join(['../data/simulations/multi_jet/figures/window6_omega_rotating_remapped_multi_jet_ures256_T1_', num2str(k) , '.mat']));

omegaw1 = data_omegaw1.omg;
omegaw2 = data_omegaw2.omg;
omegaw3 = data_omegaw3.omg;
omegaw4 = data_omegaw4.omg;
omegaw5 = data_omegaw5.omg;
% omegaw6 = data_omegaw6.u;




nexttile
imagesc(phis1,thetas1,omegaw1), hold on
xlabel('width = $2^{-3}$')
% xlabel('$\lambda$')
% ylabel('$\theta$')
set(gca,'YTickLabel',[]);
set(gca,'XTickLabel',[]);

axis square
axis tight

phi_b = linspace(phi0 - 1/16, phi0 + 1/16);
theta_l = linspace(theta0 - 1/16, theta0 + 1/16);
plot(uns*(phi0-1/16), theta_l,'w-', 'LineWidth',1), hold on
plot(uns*(phi0+1/16), theta_l,'w-', 'LineWidth',1), hold on
plot(phi_b, (theta0+1/16)*uns,'w-', 'LineWidth',1), hold on
plot(phi_b, (theta0-1/16)*uns,'w-', 'LineWidth',1), hold on



nexttile
imagesc(phis2,thetas2,omegaw2), hold on
xlabel('width = $2^{-5}$')
% xlabel('$\lambda$')
% ylabel('$\theta$')
set(gca,'YTickLabel',[]);
set(gca,'XTickLabel',[]);

axis square
axis tight

phi_b = linspace(phi0 - 2^(-6), phi0 + 2^(-6));
theta_l = linspace(theta0 - 2^(-6), theta0 + 2^(-6));
plot(uns*(phi0-2^(-6)), theta_l,'w-', 'LineWidth',1), hold on
plot(uns*(phi0+2^(-6)), theta_l,'w-', 'LineWidth',1), hold on
plot(phi_b, (theta0+2^(-6))*uns,'w-', 'LineWidth',1), hold on
plot(phi_b, (theta0-2^(-6))*uns,'w-', 'LineWidth',1), hold on



nexttile
imagesc(phis3,thetas3,omegaw3), hold on
xlabel('width = $2^{-7}$')
% xlabel('$\lambda$')
% ylabel('$\theta$')
set(gca,'YTickLabel',[]);
set(gca,'XTickLabel',[]);

axis square
axis tight

phi_b = linspace(phi0 - 2^(-8), phi0 + 2^(-8));
theta_l = linspace(theta0 - 2^(-8), theta0 + 2^(-8));
plot(uns*(phi0-2^(-8)), theta_l,'w-', 'LineWidth',1), hold on
plot(uns*(phi0+2^(-8)), theta_l,'w-', 'LineWidth',1), hold on
plot(phi_b, (theta0+2^(-8))*uns,'w-', 'LineWidth',1), hold on
plot(phi_b, (theta0-2^(-8))*uns,'w-', 'LineWidth',1), hold on



nexttile
imagesc(phis4,thetas4,omegaw4), hold on
xlabel('width = $2^{-9}$')
% xlabel('$\lambda$')
% ylabel('$\theta$')
set(gca,'YTickLabel',[]);
set(gca,'XTickLabel',[]);

axis square
axis tight

phi_b = linspace(phi0 - 2^(-10), phi0 + 2^(-10));
theta_l = linspace(theta0 - 2^(-10), theta0 + 2^(-10));
plot(uns*(phi0-2^(-10)), theta_l,'w-', 'LineWidth',1), hold on
plot(uns*(phi0+2^(-10)), theta_l,'w-', 'LineWidth',1), hold on
plot(phi_b, (theta0+2^(-10))*uns,'w-', 'LineWidth',1), hold on
plot(phi_b, (theta0-2^(-10))*uns,'w-', 'LineWidth',1), hold on



nexttile
imagesc(phis5,thetas5,omegaw5), hold on
xlabel('width = $2^{-11}$')
% xlabel('$\lambda$')
% ylabel('$\theta$')
set(gca,'YTickLabel',[]);
set(gca,'XTickLabel',[]);

axis square
axis tight


caxis manual
colormap(inferno)
bottom = min(min(omegaw1)); top = max(max(omegaw1));
caxis([bottom top]);
