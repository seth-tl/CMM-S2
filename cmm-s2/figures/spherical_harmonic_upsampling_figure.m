clear all
close all
set(0,'defaulttextinterpreter','latex');

load spine

figure()
data_omegaw1 = load("../../data/spectrum_experiment_sph_harm_upsampling256.mat").samples;
data_omegaw2 = load("../../data/spectrum_experiment_sph_harm_upsampling512.mat").samples;
data_omegaw3 = load("../../data/spectrum_experiment_sph_harm_upsampling1024.mat").samples;
data_omegaw4 = load("../../data/spectrum_experiment_sph_harm_upsampling2048.mat").samples;

tiledlayout(1,2, 'TileSpacing','none','Padding','none');

nexttile
N_pts = 256
phi_grid = linspace(0, 2*pi, 2*N_pts);
theta_grid = linspace(0, pi, N_pts+1);
[phi, theta] = meshgrid(phi_grid, theta_grid);
%[x_s,y_s,z_s]=sphere(npixels);
x_s = sin(theta).*cos(phi);
y_s = sin(theta).*sin(phi);
z_s = cos(theta);
p1 = surf(x_s,y_s,z_s, data_omegaw1); hold on
% alpha(p1,0.7)
text(0,0,1.4, '$L = 256$', 'FontSize', 14)

% shading flat;
shading interp
colormap(inferno);
% spherefun.plotEarth('k-');

axis tight
axis off


nexttile
N_pts = 512
phi_grid = linspace(0, 2*pi, 2*N_pts);
theta_grid = linspace(0, pi, N_pts+1);
[phi, theta] = meshgrid(phi_grid, theta_grid);
%[x_s,y_s,z_s]=sphere(npixels);
x_s = sin(theta).*cos(phi);
y_s = sin(theta).*sin(phi);
z_s = cos(theta);
p1 = surf(x_s,y_s,z_s, data_omegaw2); hold on
% alpha(p1,0.7)
% title('$L = 512$', 'Position','South')
text(0,0, 1.4, '$L = 512$', 'FontSize', 14)
% shading flat;
shading interp
colormap(inferno);
% spherefun.plotEarth('k-');

axis tight
axis off

figure()
tiledlayout(1,2, 'TileSpacing','none','Padding','none');


nexttile
N_pts = 1024
phi_grid = linspace(0, 2*pi, 2*N_pts);
theta_grid = linspace(0, pi, N_pts+1);
[phi, theta] = meshgrid(phi_grid, theta_grid);
%[x_s,y_s,z_s]=sphere(npixels);
x_s = sin(theta).*cos(phi);
y_s = sin(theta).*sin(phi);
z_s = cos(theta);
p1 = surf(x_s,y_s,z_s, data_omegaw3); hold on
% alpha(p1,0.7)
text(0,0,1.4, '$L = 1024$', 'FontSize', 14)

% shading flat;
shading interp
colormap(inferno);
% spherefun.plotEarth('k-');

axis tight
axis off


nexttile
N_pts = 2048
phi_grid = linspace(0, 2*pi, 2*N_pts);
theta_grid = linspace(0, pi, N_pts+1);
[phi, theta] = meshgrid(phi_grid, theta_grid);
%[x_s,y_s,z_s]=sphere(npixels);
x_s = sin(theta).*cos(phi);
y_s = sin(theta).*sin(phi);
z_s = cos(theta);
p1 = surf(x_s,y_s,z_s, data_omegaw4); hold on
% alpha(p1,0.7)
text(0,0,1.4, '$L = 2048$', 'FontSize', 14)

% shading flat;
shading interp
colormap(inferno);
% spherefun.plotEarth('k-');

axis tight
axis off