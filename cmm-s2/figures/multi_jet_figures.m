clear all
close all
set(0,'defaulttextinterpreter','latex');

load spine

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


