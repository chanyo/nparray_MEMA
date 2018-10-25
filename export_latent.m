clear all;
clc;
close all;

% load embedding
rng('default');

FLAG_EMBEDDING = 1;
output_dir = './output/';


dim_latent = 16;
load(sprintf('result_VAE_LINCS_196_organization_d%d.mat', dim_latent));

meta_fname = textread('list_train_HCC1143_organization.txt','%s');

meta = readtable('hcc1143_low_serum_imageIDs.csv');

ligand_lable = unique(meta.Ligand);

for i=1:length(meta_fname)
    id = []; id = strfind(meta_fname{i}, '/');
    fname{i} = meta_fname{i}(id(end)+1:end-4);
    
    L(i) = find(strcmp([ligand_lable], meta.Ligand(find(meta.ImageID == str2num(fname{i}))))==1);
end



filename = fname';
T = table(filename, x_train_encoded);

writetable(T, sprintf('export_latent_d%d.csv', dim_latent));







x = fast_tsne(x_train_encoded, 2, [], 20,0.7); % 0.2


x = bsxfun(@minus, x, min(x));
x = bsxfun(@rdivide, x, max(x));

gscatter(x(:,1),x(:,2), L);

%%
cmap = colormap(jet(64));
for i=1:max(L)
    id = []; id = find(L==i);
    plot(x(id,1),x(id,2), '.', 'Color', cmap(i,:));
    hold on;
end





%%
%[coeff, score, latent] = pca(x_train_encoded);
%gscatter(score(:,1), score(:,2), L);
%biplot(coeff(:,1:2) , 'scores', score(:,1:2), 'varlabels', L);

%imwrite(G, 'VAE_cnn_HCC1143_grid.png', 'png');

%%
N_cluster = dim_latent;

% KSSC
Z_kssc = kssc_exact_par(x_train_encoded', 0.1, 200);
[kssc_clusters,NcutEigenvectors,NcutEigenvalues] = ncutW((abs(Z_kssc)+abs(Z_kssc')), N_cluster);

Label_SSC = condense_clusters(kssc_clusters,1);


kssc_sorted = []; NcutEigv_sorted = [];
for i=1:N_cluster
    ii = find(Label_SSC == i);
    kssc_sorted = [kssc_sorted; kssc_clusters(ii,:)];
    NcutEigv_sorted = [NcutEigv_sorted; NcutEigenvectors(ii,:)];
end


figure
imagesc([kssc_clusters kssc_sorted]);
colormap redbluecmap

 
y = fast_tsne(NcutEigenvectors,2,[], 20, 0.7);
y = bsxfun(@minus, y, min(y));
y = bsxfun(@rdivide, y, max(y));

    
%
figure
cmap = colormap(jet(64));
for i=1:max(L)
    id = []; id = find(L==i);
    plot(y(id,1),y(id,2), 'ok', 'MarkerFaceColor', cmap(i,:),'MarkerSize',5);
    
    legend_name{i} = sprintf('%s', ligand_lable{i});
    hold on;
end
view(90,90);
legend(legend_name);
saveas(gcf, sprintf('SSC_tsne_d%d.fig', dim_latent));
saveas(gcf, sprintf('SSC_tsne_d%d.png', dim_latent),'png');


figure
cmap = colormap(jet(64));
for i=1:max(L)
    id = []; id = find(L==i);
    plot(y(id,1),y(id,2), 'ok', 'MarkerFaceColor', cmap(i,:),'MarkerSize',5);
    
    legend_name{i} = sprintf('%s', ligand_lable{i});
    hold on;
end



figure
cmap = colormap(jet(64));
for i=1:max(L)
    subplot(7, ceil(max(L)/7), i);
    
        
    id = find(L~= i);
    plot(x(id,1),x(id,2), '.', 'Color', [211 211 211]/255,  'MarkerSize',5);
    
    hold on,
    
    
    id = []; id = find(L==i);
    plot(x(id,1),x(id,2), '.', 'Color', cmap(i,:),'MarkerSize',5);
    title(legend_name{i})
    hold on;

    view(90,90);
    
end

legend(legend_name);
saveas(gcf, sprintf('org_tsne_d%d.fig', dim_latent));
saveas(gcf, sprintf('org_tsne_d%d.png', dim_latent),'png');

%plot(NcutEigv_sorted)
%imagesc([kssc_clusters kssc_sorted]);


% scalable SSC
%Label = get_sssc_clusters(Xn', floor(size(Xn,1)*0.2), 0.1, N_cluster);



%% load validation image filenames
x = y; % USE SSC result



if FLAG_EMBEDDING
    for i=1:length(meta_fname)
        id = []; id = strfind(meta_fname{i}, '/');
        fs{i} = sprintf('%s%s.png', output_dir, meta_fname{i}(id(end)+1:end-4));
    end

    N = length(fs);



%% create an embedding image



    S = N; % size of full embedding image
    G = zeros(S, S, 3, 'uint8');
    s = 120; % size of every single image

    Ntake = N;
    for i=1:Ntake

        if mod(i, 100)==0
            fprintf('%d/%d...\n', i, Ntake);
        end

        % location
        a = ceil(x(i, 1) * (S-s)+1);
        b = ceil(x(i, 2) * (S-s)+1);
        a = a-mod(a-1,s)+1;
        b = b-mod(b-1,s)+1;
        if G(a,b,1) ~= 0
            continue % spot already filled
        end

        I = imread(fs{i});
        if size(I,3)==1, I = cat(3,I,I,I); end
        I = imresize(I, [s, s]);
        for j=1:3
            I(:,:,j) = imadjust(I(:,:,j));
        end



        G(a:a+s-1, b:b+s-1, :) = I;

    end
    figure
    imshow(G);

    %
    imwrite(G, sprintf('VAE_cnn_HCC1143_d%d.png', dim_latent), 'png');
end

if FLAG_EMBEDDING
    %% do a guaranteed quade grid layout by taking nearest neighbor

    S = 7055; % size of final image
    G = zeros(S, S, 3, 'uint8');
    s = 120; % size of every image thumbnail

    xnum = S/s;
    ynum = S/s;
    used = false(N, 1);

    qq=length(1:s:S);
    abes = zeros(qq*2,2);
    i=1;
    for a=1:s:S
        for b=1:s:S
            abes(i,:) = [a,b];
            i=i+1;
        end
    end
    %abes = abes(randperm(size(abes,1)),:); % randperm

    for i=1:size(abes,1)
        a = abes(i,1);
        b = abes(i,2);
        %xf = ((a-1)/S - 0.5)/2 + 0.5; % zooming into middle a bit
        %yf = ((b-1)/S - 0.5)/2 + 0.5;
        xf = (a-1)/S;
        yf = (b-1)/S;
        dd = sum(bsxfun(@minus, x(:,1:2), [xf, yf]).^2,2);
        dd(used) = inf; % dont pick these
        [dv,di] = min(dd); % find nearest image

        used(di) = true; % mark as done
        I = imread(fs{di});
        if size(I,3)==1, I = cat(3,I,I,I); end
        I = imresize(I, [s, s]);

        for j=1:3
            I(:,:,j) = imadjust(I(:,:,j));
        end



        G(a:a+s-1, b:b+s-1, :) = I;

        if mod(i,100)==0
            fprintf('%d/%d\n', i, size(abes,1));
        end
    end
    figure
    imshow(G);

    %
    imwrite(G, sprintf( 'VAE_cnn_HCC1143_grid_d%d.png',dim_latent), 'png');
end