clear all
close all
data = load('spline_interpolant.mat')

u_num = reshape(data.u_num, [512,512]);
u_true = reshape(data.u_true, [512,512]);

fig = figure()
tiledlayout(1,3,"TileSpacing","tight", "Padding","tight")

phis = linspace(0,2*pi,512);
thetas = linspace(0,pi,512);


nexttile
imagesc(phis,thetas, u_true), hold on
colormap(inferno)

nexttile
imagesc(phis,thetas, u_num), hold on
colormap(inferno)

nexttile
imagesc(phis,thetas, abs(u_num-u_true)), hold on
colormap(inferno)
colorbar()


fig = figure()
tiledlayout(2,3,"TileSpacing","tight", "Padding","tight")

phis = linspace(0,2*pi,512);
thetas = linspace(0,pi,512);

u_grad_true_x = reshape(data.u_grad_true(1,:), [512, 512]);
u_grad_true_y = reshape(data.u_grad_true(2,:), [512, 512]);
u_grad_true_z = reshape(data.u_grad_true(3,:), [512, 512]);

u_grad_num_x = reshape(data.u_grad_num(1,:), [512, 512]);
u_grad_num_y = reshape(data.u_grad_num(2,:), [512, 512]);
u_grad_num_z = reshape(data.u_grad_num(3,:), [512, 512]);



nexttile
imagesc(phis,thetas, u_grad_true_x), hold on
colormap(inferno)

nexttile
imagesc(phis,thetas, u_grad_true_y), hold on
colormap(inferno)

nexttile
imagesc(phis,thetas, u_grad_true_z), hold on
colormap(inferno)


nexttile
imagesc(phis,thetas, u_grad_num_x), hold on
colormap(inferno)

nexttile
imagesc(phis,thetas, u_grad_num_y), hold on
colormap(inferno)

nexttile
imagesc(phis,thetas, u_grad_num_z), hold on
colormap(inferno)
