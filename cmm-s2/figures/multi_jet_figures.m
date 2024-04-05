clear all
close all
set(0,'defaulttextinterpreter','latex');

load spine
% v = VideoWriter('../EulerScripts/data/videos/remapped_perturbed_zonal_jet_T1.avi', 'Motion JPEG AVI');
% open(v);

lat = 0;
lon = 10;
% cam_rot_rate = 0.1
%
N_pts = 1000
phi_grid = linspace(0, 2*pi, 2*N_pts);
theta_grid = linspace(0, pi, N_pts+1);
[phi, theta] = meshgrid(phi_grid, theta_grid);
%[x_s,y_s,z_s]=sphere(npixels);
x_s = sin(theta).*cos(phi);
y_s = sin(theta).*sin(phi);
z_s = cos(theta);



j = 0
figure()
tiledlayout(3,3, 'TileSpacing','none','Padding','none');

omega = load('../../data/multi_jet_experiment_voriticity_figures.mat').omgs;

nexttile
% figure()
p1 = surf(x_s,y_s,z_s, squeeze(omega(1,:,:))); hold on
alpha(p1,0.9)
view([lon lat])

% shading flat;
shading interp
colormap(inferno);
% spherefun.plotEarth('k-');
axis square
axis tight
axis off
% title('t= T/10')

nexttile
% figure()
p1 = surf(x_s,y_s,z_s, squeeze(omega(2,:,:))); hold on
alpha(p1,0.9)
view([lon lat])

% shading flat;
shading interp
colormap(inferno);
% spherefun.plotEarth('k-');
axis square
axis tight
axis off
% title('t= 2T/10')

nexttile
% figure()
p1 = surf(x_s,y_s,z_s, squeeze(omega(3,:,:))); hold on
alpha(p1,0.9)
view([lon lat])

% shading flat;
shading interp
colormap(inferno);
% spherefun.plotEarth('k-');
axis square
axis tight
axis off
% title('t= 3T/10')


% figure()
% tiledlayout(1,3, 'TileSpacing','none','Padding','none');
% 

nexttile
% figure()
p1 = surf(x_s,y_s,z_s,squeeze(omega(4,:,:))); hold on
alpha(p1,0.9)
view([lon lat])

% shading flat;
shading interp
colormap(inferno);
% spherefun.plotEarth('k-');
axis square
axis tight
axis off
% title('t= 4T/10')

nexttile
% figure()
p1 = surf(x_s,y_s,z_s, squeeze(omega(5,:,:))); hold on
alpha(p1,0.9)
view([lon lat])

% shading flat;
shading interp
colormap(inferno);
% spherefun.plotEarth('k-');
axis square
axis tight
axis off
% title('t= 5T/10'
nexttile
% figure()
p1 = surf(x_s,y_s,z_s, squeeze(omega(6,:,:))); hold on
alpha(p1,0.9)
view([lon lat])

% shading flat;
shading interp
colormap(inferno);
% spherefun.plotEarth('k-');
axis square
axis tight
axis off
% title('t= 6T/10')

% figure()
% tiledlayout(1,3, 'TileSpacing','none','Padding','none');

nexttile
% figure()
p1 = surf(x_s,y_s,z_s, squeeze(omega(7,:,:))); hold on
alpha(p1,0.9)
view([lon lat])

% shading flat;
shading interp
colormap(inferno);
% spherefun.plotEarth('k-');
axis square
axis tight
axis off
% title('t= 7T/10')


nexttile
% figure()
p1 = surf(x_s,y_s,z_s, squeeze(omega(8,:,:))); hold on
alpha(p1,0.9)
view([lon lat])

% shading flat;
shading interp
colormap(inferno);
% spherefun.plotEarth('k-');
axis square
axis tight
axis off

% title('t= 8T/10')


nexttile
% figure()
p1 = surf(x_s,y_s,z_s, squeeze(omega(9,:,:))); hold on
alpha(p1,0.9)
view([lon lat])

% shading flat;
shading interp
colormap(inferno);
% spherefun.plotEarth('k-');
axis square
axis tight
axis off
% title('t= 9T/10')


% data_omega = load(join(['../data/simulations/multi_jet/figures/global_omega_rotating_remapped_multi_jet_ures256_T1_final_hires.mat']));
% omega_g = data_omega.glob;
%
%
% N_pts = 2500
% phi_grid = linspace(0, 2*pi, N_pts);
% theta_grid = linspace(0, pi, N_pts);
% [phi, theta] = meshgrid(phi_grid, theta_grid);
% %[x_s,y_s,z_s]=sphere(npixels);
% x_s = sin(theta).*cos(phi);
% y_s = sin(theta).*sin(phi);
% z_s = cos(theta);
%
% figure()
% imagesc(phi_grid,theta_grid,omega_g), hold on
% colormap(inferno)
% % colormap(jet(1024))

%
% % % omega_w1 = data_omegaW.window1; omega_w2 = data_omegaW.window2;
% phi_b = linspace(phi0 - 1/4, phi0 + 1/4);
% theta_l = linspace(theta0 - 1/4, theta0 + 1/4);
%
%
% plot(ones*(phi0-1/4), theta_l,'w-', 'LineWidth',1.8), hold on
% plot(ones*(phi0+1/4), theta_l,'w-', 'LineWidth',1.8), hold on
% plot(phi_b, (theta0+1/4)*ones,'w-', 'LineWidth',1.8), hold on
% plot(phi_b, (theta0-1/4)*ones,'w-', 'LineWidth',1.8), hold on
%
%
%
% xlabel('$\lambda$')
% ylabel('$\theta$')
% axis square
% axis tight
% title('t=T')
%
%
% % Zoom windows

% phi0 = 3.36815; theta0 = 0.854678;
% phi0 = 3.23835; theta0 = 1.06811;
phi0 = 3.22055; theta0 = 1.1963;


phis2 = linspace(phi0- 2^-1, phi0 + 2^-1,N_pts);
thetas2 = linspace(theta0- 2^-1,theta0 + 2^-1, N_pts);
[X2,Y2] = meshgrid(phis2, thetas2);



phis3 = linspace(phi0- 2^-3, phi0 + 2^-3,N_pts);
thetas3 = linspace(theta0- 2^-3,theta0 + 2^-3, N_pts);
[X3,Y3] = meshgrid(phis3, thetas3);

phis4 = linspace(phi0- 2^-6, phi0 + 2^-6,N_pts);
thetas4 = linspace(theta0- 2^-6,theta0 + 2^-6, N_pts);
[X4,Y4] = meshgrid(phis4, thetas4);

phis5 = linspace(phi0- 2^-9, phi0 + 2^-9, N_pts);
thetas5 = linspace(theta0- 2^-9,theta0 + 2^-9, N_pts);
[X5,Y5] = meshgrid(phis5, thetas5);

phis6 = linspace(phi0- 2^-12, phi0 + 2^-12,N_pts);
thetas6 = linspace(theta0- 2^-12,theta0 + 2^-12, N_pts);
[X6,Y6] = meshgrid(phis6, thetas6);




% j = 0
% figure()
% tiledlayout(2,3, 'TileSpacing','compact','Padding','none');

% %
% data_omegaw0 = load('../data/simulations/multi_jet/figures/global_omega_redo_rotating_remapped_multi_jet_ures256_T10_window_initial.mat');
% data_omegaw1 = load('../data/simulations/multi_jet/figures/global_omega_redo_rotating_remapped_multi_jet_ures256_T10_window_0.mat');
% data_omegaw2 = load('../data/simulations/multi_jet/figures/global_omega_redo_rotating_remapped_multi_jet_ures256_T10_window_1.mat')
% data_omegaw3 = load('../data/simulations/multi_jet/figures/global_omega_redo_rotating_remapped_multi_jet_ures256_T10_window_2.mat')
% data_omegaw4 = load('../data/simulations/multi_jet/figures/global_omega_redo_rotating_remapped_multi_jet_ures256_T10_window_3.mat')
% data_omegaw5 = load('../data/simulations/multi_jet/figures/global_omega_redo_rotating_remapped_multi_jet_ures256_T10_window_4.mat')

% omega_g = data_omegaw0.omg;
% omegaw2 = data_omegaw1.omg;
% omegaw3 = data_omegaw2.omg;
% omegaw4 = data_omegaw3.omg;
% omegaw5 = data_omegaw4.omg;
% omegaw6 = data_omegaw5.omg;


% %

% nexttile
% imagesc(phi_grid,theta_grid,omega_g), hold on
% % xlabel('width = $2^{-1}$')
% xlabel('$\lambda$')
% ylabel('$\theta$')
% colormap(inferno)

% % title('t = T')
% % set(gca,'YTickLabel',[]);
% % set(gca,'XTickLabel',[]);

% axis square
% axis tight

% phi_b = linspace(phi0 - 1/2, phi0 + 1/2);
% theta_l = linspace(theta0 - 1/2, theta0 + 1/2);
% ones = ones(length(theta_l));

% plot(ones*(phi0-1/2), theta_l,'-w', 'LineWidth',1), hold on
% plot(ones*(phi0+1/2), theta_l,'-w', 'LineWidth',1), hold on
% plot(phi_b, (theta0+1/2)*ones,'-w', 'LineWidth',1), hold on
% plot(phi_b, (theta0-1/2)*ones,'-w', 'LineWidth',1), hold on



% nexttile
% imagesc(phis2,thetas2,omegaw2), hold on
% xlabel('width = $2^{-1}$')
% colormap(inferno)

% % xlabel('$\lambda$')
% % ylabel('$\theta$')
% set(gca,'YTickLabel',[]);
% set(gca,'XTickLabel',[]);

% axis square
% axis tight

% phi_b = linspace(phi0 - 2^(-4), phi0 + 2^(-4));
% theta_l = linspace(theta0 - 2^(-4), theta0 + 2^(-4));
% plot(ones*(phi0-2^(-4)), theta_l,'-w', 'LineWidth',1), hold on
% plot(ones*(phi0+2^(-4)), theta_l,'-w', 'LineWidth',1), hold on
% plot(phi_b, (theta0+2^(-4))*ones,'-w', 'LineWidth',1), hold on
% plot(phi_b, (theta0-2^(-4))*ones,'-w', 'LineWidth',1), hold on



% nexttile
% imagesc(phis3,thetas3,omegaw3), hold on
% xlabel('width = $2^{-3}$')
% colormap(inferno)

% % xlabel('$\lambda$')
% % ylabel('$\theta$')
% set(gca,'YTickLabel',[]);
% set(gca,'XTickLabel',[]);

% axis square
% axis tight

% phi_b = linspace(phi0 - 2^(-6), phi0 + 2^(-6));
% theta_l = linspace(theta0 - 2^(-6), theta0 + 2^(-6));
% plot(ones*(phi0-2^(-6)), theta_l,'-w', 'LineWidth',1), hold on
% plot(ones*(phi0+2^(-6)), theta_l,'-w', 'LineWidth',1), hold on
% plot(phi_b, (theta0+2^(-6))*ones,'-w', 'LineWidth',1), hold on
% plot(phi_b, (theta0-2^(-6))*ones,'-w', 'LineWidth',1), hold on



% nexttile
% imagesc(phis4,thetas4,omegaw4), hold on
% xlabel('width = $2^{-6}$')
% colormap(inferno)

% % xlabel('$\lambda$')
% % ylabel('$\theta$')
% set(gca,'YTickLabel',[]);
% set(gca,'XTickLabel',[]);

% axis square
% axis tight

% phi_b = linspace(phi0 - 2^(-9), phi0 + 2^(-9));
% theta_l = linspace(theta0 - 2^(-9), theta0 + 2^(-9));
% plot(ones*(phi0-2^(-9)), theta_l,'-w', 'LineWidth',1), hold on
% plot(ones*(phi0+2^(-9)), theta_l,'-w', 'LineWidth',1), hold on
% plot(phi_b, (theta0+2^(-9))*ones,'-w', 'LineWidth',1), hold on
% plot(phi_b, (theta0-2^(-9))*ones,'-w', 'LineWidth',1), hold on



% nexttile
% imagesc(phis5,thetas5,omegaw5), hold on
% xlabel('width = $2^{-9}$')
% colormap(inferno)

% % xlabel('$\lambda$')
% % ylabel('$\theta$')
% set(gca,'YTickLabel',[]);
% set(gca,'XTickLabel',[]);

% axis square
% axis tight


% phi_b = linspace(phi0 - 2^(-12), phi0 + 2^(-12));
% theta_l = linspace(theta0 - 2^(-12), theta0 + 2^(-12));
% plot(ones*(phi0-2^(-12)), theta_l,'-w', 'LineWidth',1), hold on
% plot(ones*(phi0+2^(-12)), theta_l,'-w', 'LineWidth',1), hold on
% plot(phi_b, (theta0+2^(-12))*ones,'-w', 'LineWidth',1), hold on
% plot(phi_b, (theta0-2^(-12))*ones,'-w', 'LineWidth',1), hold on




% nexttile
% imagesc(phis6,thetas6,omegaw6)
% xlabel('width = $2^{-12}$')
% colormap(inferno)

% % xlabel('$\lambda$')
% % ylabel('$\theta$')
% set(gca,'YTickLabel',[]);
% set(gca,'XTickLabel',[]);

% axis square
% axis tight


% caxis manual
% colormap(inferno)
% bottom = min(min(omega_g)); top = max(max(omega_g));
% caxis([bottom top]);


% %
% % figure()
% % imagesc(phis3,thetas3,omegaw3), hold on
% % colormap(inferno)
% % % xlabel('$\lambda$')
% % % ylabel('$\theta$')
% % set(gca,'YTickLabel',[]);
% % set(gca,'XTickLabel',[]);
% %
% % axis square
% % axis tight
