figure()
tiledlayout(2,3, 'TileSpacing','compact','Padding','none');


N_pts = 1500
phi_grid = linspace(0, 2*pi, N_pts);
theta_grid = linspace(0, pi, N_pts);

omega = load('../../data/multi_jet_experiment_voriticity_figures.mat').omgs;
omega = squeeze(omega(10,:,:));


nexttile
imagesc(phi_grid,theta_grid,omega), hold on
colormap(inferno)
%
% omega_w1 = data_omegaW.window1; omega_w2 = data_omegaW.window2;
phi0 = 3.22055;  theta0 = 1.1963;


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


% Zoom windows


N_pts = 1000
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



data = load("../../data/multi_jet_zoom_figures.mat").omgs;




nexttile
imagesc(phis1,thetas1,squeeze(data(1,:,:))), hold on
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
imagesc(phis2,thetas2,squeeze(data(2,:,:))), hold on
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
imagesc(phis3,thetas3,squeeze(data(3,:,:))), hold on
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
imagesc(phis4,thetas4,squeeze(data(4,:,:))), hold on
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
imagesc(phis5,thetas5,squeeze(data(5,:,:))), hold on
xlabel('width = $2^{-11}$')
% xlabel('$\lambda$')
% ylabel('$\theta$')
set(gca,'YTickLabel',[]);
set(gca,'XTickLabel',[]);

axis square
axis tight


caxis manual
colormap(inferno)
bottom = min(min(squeeze(data(1,:,:)))); top = max(max(squeeze(data(1,:,:))));
caxis([bottom top]);