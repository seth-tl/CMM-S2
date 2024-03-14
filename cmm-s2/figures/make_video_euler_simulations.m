clear all
close all
set(0,'defaulttextinterpreter','latex');

load spine
v = VideoWriter('../data/simulations/videos/turbulent_rhwave_static_T100.avi', 'Motion JPEG AVI');
open(v);

% lat = 10;
% lon = 30;
% cam_rot_rate = 0.1

N_pts = 256
phi_grid = linspace(0, 2*pi, 2*N_pts);
theta_grid = linspace(0, pi, N_pts+1);
[phi, theta] = meshgrid(phi_grid, theta_grid);
%[x_s,y_s,z_s]=sphere(npixels);
x_s = sin(theta).*cos(phi);
y_s = sin(theta).*sin(phi);
z_s = cos(theta);

lls = linspace(1,1000, 1000);


f = figure();
f.WindowState = 'maximized';
set(f,'position',[.05 .05 .9 .9])

for k = 0:99
   for i = 1:5
      data = load(join(['../data/simulations/spectra/random_vorticity_spectra/sampling_omega_static_long_term_perturbed_RHwave_tscl10000_ures256_T100_L1000_', num2str(k) , '.mat']));
      % data = load(join(['../data/simulations/spectra/random_vorticity_spectra/omega_rotating_long_term_perturbed_RHwave_ures256_T100_', num2str(k) , '.mat']));

      omega = data.omg;

      Handle(1,1) = subplot(1,2,1, 'replace')
      % p1 = surf(x_s,y_s,z_s, omega); hold on
      imagesc(omega); hold on
      shading interp
      colormap(inferno);
  %     spherefun.plotEarth('k-');
      axis square
      axis tight
      axis off
      % view([lon,lat])
      title(join(['t = ' , num2str(k+1)]),"FontSize",14)

      Handle(1,2) = subplot(1,2,2, 'replace')
      title(join(['t = ' , num2str(k+1)]))

      data = load(join(['../data/simulations/spectra/random_vorticity_spectra/spectrum_static_long_term_perturbed_RHwave_tscl10000_ures256_T100_L1000_', num2str(k) , '.mat']));
      % data = load(join(['../../casimirs_experiment/data/simulations/perturbed_zonal_jet/spectra/spectrum_perturbed_zonal_jet_id_9_tscl10000_ures256_T100_L1000_', num2str(k) , '.mat']));


      spectrum = data.ells;

      p = loglog(lls, spectrum(1,:), '-k');  hold on;
      %   set(gca, 'YScale', 'log')
      ylabel('Energy')
      xlabel('$\ell$')
      grid on
      axis tight

      plot(lls, 0.0001*lls.^(-1), '--r'), hold on
      plot(lls, 20*lls.^(-3), '--b'), hold on
      plot(lls, 1000*lls.^(-5), '--g'), hold on
      title(join(['t = ' , num2str(k+1)]),"FontSize",14)

      % set(Handle(1,1:2), 'View', [lon + cam_rot_rate*k,lat])

      frame = getframe(gcf);
      writeVideo(v,frame);
   end
end

close(v);


% figure()
% k = 0
% data = load(join(['../data/simulations/spectra/random_vorticity_spectra/sampling_omega_static_long_term_perturbed_RHwave_tscl10000_ures256_T100_L1000_', num2str(k) , '.mat']));

% % tracer = load(file).tracer_T;
% omega = data.omg;

% Handle(1,1) = subplot(1,2,1, 'replace')
% % p1 = surf(x_s,y_s,z_s, omega); hold on
% imagesc(omega); hold on
% shading interp
% colormap(inferno);
% %     spherefun.plotEarth('k-');
% axis square
% axis tight
% axis off
% % view([lon,lat])
% title(join(['t = ' , num2str(k+1)]),"FontSize",14)

% Handle(1,2) = subplot(1,2,2, 'replace')
% title(join(['t = ' , num2str(k+1)]))

% data = load(join(['../data/simulations/spectra/random_vorticity_spectra/spectrum_static_long_term_perturbed_RHwave_tscl10000_ures256_T100_L1000_', num2str(k) , '.mat']));
% % data = load(join(['../../casimirs_experiment/data/simulations/perturbed_zonal_jet/spectra/spectrum_perturbed_zonal_jet_id_9_tscl10000_ures256_T100_L1000_', num2str(k) , '.mat']));


% spectrum = data.ells;

% p = loglog(lls, spectrum(1,:), '-k');  hold on;
% %   set(gca, 'YScale', 'log')
% ylabel('Energy')
% xlabel('$\ell$')
% grid on
% axis tight

% % plot(lls, 0.0001*lls.^(-1), '--r'), hold on
% % plot(lls, 20*lls.^(-3), '--b'), hold on
% % plot(lls, 1000*lls.^(-5), '--g'), hold on
% % title(join(['t = ' , num2str(k)]),"FontSize",14)


% figure()
% k = 20
% data = load(join(['../data/simulations/spectra/random_vorticity_spectra/sampling_omega_static_long_term_perturbed_RHwave_tscl10000_ures256_T100_L1000_', num2str(k) , '.mat']));

% % tracer = load(file).tracer_T;
% omega = data.omg;

% Handle(1,1) = subplot(1,2,1, 'replace')
% % p1 = surf(x_s,y_s,z_s, omega); hold on
% imagesc(omega); hold on
% shading interp
% colormap(inferno);
% %     spherefun.plotEarth('k-');
% axis square
% axis tight
% axis off
% % view([lon,lat])
% title(join(['t = ' , num2str(k+1)]),"FontSize",14)

% Handle(1,2) = subplot(1,2,2, 'replace')
% title(join(['t = ' , num2str(k+1)]))

% data = load(join(['../data/simulations/spectra/random_vorticity_spectra/spectrum_static_long_term_perturbed_RHwave_tscl10000_ures256_T100_L1000_', num2str(k) , '.mat']));
% % data = load(join(['../../casimirs_experiment/data/simulations/perturbed_zonal_jet/spectra/spectrum_perturbed_zonal_jet_id_9_tscl10000_ures256_T100_L1000_', num2str(k) , '.mat']));


% spectrum = data.ells;

% p = loglog(lls, spectrum(1,:), '-k');  hold on;
% %   set(gca, 'YScale', 'log')
% ylabel('Energy')
% xlabel('$\ell$')
% grid on
% axis tight

% % plot(lls, 0.0001*lls.^(-1), '--r'), hold on
% % plot(lls, 20*lls.^(-3), '--b'), hold on
% % plot(lls, 1000*lls.^(-5), '--g'), hold on
% % title(join(['t = ' , num2str(k)]),"FontSize",14)


% figure()
% k = 99
% data = load(join(['../data/simulations/spectra/random_vorticity_spectra/sampling_omega_static_long_term_perturbed_RHwave_tscl10000_ures256_T100_L1000_', num2str(k) , '.mat']));

% % tracer = load(file).tracer_T;
% omega = data.omg;

% Handle(1,1) = subplot(1,2,1, 'replace')
% % p1 = surf(x_s,y_s,z_s, omega); hold on
% imagesc(omega); hold on
% shading interp
% colormap(inferno);
% %     spherefun.plotEarth('k-');
% axis square
% axis tight
% axis off
% % view([lon,lat])
% title(join(['t = ' , num2str(k+1)]),"FontSize",14)

% Handle(1,2) = subplot(1,2,2, 'replace')
% title(join(['t = ' , num2str(k+1)]))

% data = load(join(['../data/simulations/spectra/random_vorticity_spectra/spectrum_static_long_term_perturbed_RHwave_tscl10000_ures256_T100_L1000_', num2str(k) , '.mat']));
% % data = load(join(['../../casimirs_experiment/data/simulations/perturbed_zonal_jet/spectra/spectrum_perturbed_zonal_jet_id_9_tscl10000_ures256_T100_L1000_', num2str(k) , '.mat']));


% spectrum = data.ells;

% p = loglog(lls, spectrum(1,:), '-k');  hold on;
% %   set(gca, 'YScale', 'log')
% ylabel('Energy')
% xlabel('$\ell$')
% grid on
% axis tight