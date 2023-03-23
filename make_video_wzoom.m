clear all
close all
set(0,'defaulttextinterpreter','latex');

load spine
% v = VideoWriter('../EulerScripts/data/videos/remapped_perturbed_zonal_jet_T1.avi', 'Motion JPEG AVI');
% open(v);

lat = 30;
lon = 10;
cam_rot_rate = 0.1;

N_pts = 2000
phi_grid = linspace(0, 2*pi, N_pts);
theta_grid = linspace(0, pi, N_pts);
[phi, theta] = meshgrid(phi_grid, theta_grid);
%[x_s,y_s,z_s]=sphere(npixels);
x_s = sin(theta).*cos(phi);
y_s = sin(theta).*sin(phi);
z_s = cos(theta);


%% window1

phis1 = linspace(pi- 2^-3, pi + 2^-3,1000);
thetas1 = linspace(pi/4 - 2^-3, pi/4 + 2^-3, 1000);
[X1,Y1] = meshgrid(phis1, thetas1);

%%window2

phis2 = linspace(pi- 2^-8, pi + 2^-8,1000);
thetas2 = linspace(pi/4 - 2^-8, pi/4 + 2^-8, 1000);
[X2,Y2] = meshgrid(phis2, thetas2);

f = figure();
f.WindowState = 'maximized';

j = 0
for k = 5:5:1000

  % data_tr = load(join(['../data/simulations/multi_jet/advected_quantities/passive_tracer_rotating_remapped_single_jet_ures256_T2_', num2str(k) , '.mat']));
  data_omega = load(join(['../data/simulations/random_vorticity/advected_quantities/omega_rotating_remapped_random_vorticity_ures256_T4_', num2str(k) , '.mat']));
  % data_omegaW = load(join(['../EulerScripts/data/simulations/multi_jet/advected_quantities/window_omega_remapped_multi_jet_ures256_T1_', num2str(k) , '.mat']));

  % tracer_g = data_tr.glob; %tracer_w1 = data_tr.window1; tracer_w2 = data_tr.window2;
  omega_g = data_omega.glob;
  % omega_w1 = data_omegaW.window1; omega_w2 = data_omegaW.window2;
  
  imagesc(phi_grid,theta_grid,omega_g)
  colormap(flipud(magma))
  xlabel('$\lambda$')
  ylabel('$\theta$')
  axis square
  axis tight

  % p1 = surf(x_s,y_s,z_s, omega_g); hold on
  % % shading flat;
  % shading interp
  % % alpha(p1, 0.7);
  % % spherefun.plotEarth('k-');
  % colormap(magma);
  % %set the axis
  % axis('square');
  % set(gca, 'Color', 'none')
  % xlabel('x')
  % ylabel('y')
  % zlabel('z')


  % j = j + 1
  saveas(f, join(['../data/videos/rotating_video/image_', num2str(j) , '.png']))


  %
  % frame = getframe(gcf);
  % writeVideo(v,frame);

end

% close(v);
