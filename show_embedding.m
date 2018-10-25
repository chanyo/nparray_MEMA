clear all
close all;
clc

% load embedding
rng('default');

data = load('result_VAE_LINCS_196_organization.mat');
output_dir = './output/';
%x = fast_tsne(data.x_train_encoded, 3, 10, 10,0.2); % 0.2
%x = fast_tsne(data.x_train_encoded, 2, 10, 10,0.1); % 0.2
x = fast_tsne(data.x_train_encoded, 3, 10, 100,0.1); % 0.2

%load('imagenet_val_embed.mat'); % load x (the embedding 2d locations from tsne)
x = bsxfun(@minus, x, min(x));
x = bsxfun(@rdivide, x, max(x));


% load validation image filenames

fs_raw = textread('list_train_HCC1143_organization.txt', '%s');

for i=1:length(fs_raw)
    id = []; id = strfind(fs_raw{i}, '/');
    fs{i} = sprintf('%s%s.png', output_dir, fs_raw{i}(id(end)+1:end-4));
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
    
   
    G(a:a+s-1, b:b+s-1, :) = I;
    
end
figure
imshow(G);

%
imwrite(G, 'VAE_cnn_HCC1143.png', 'png');


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

    
   
    
    G(a:a+s-1, b:b+s-1, :) = I;
    
    if mod(i,100)==0
        fprintf('%d/%d\n', i, size(abes,1));
    end
end
figure
imshow(G);

%
imwrite(G, 'VAE_cnn_HCC1143_grid.png', 'png');

