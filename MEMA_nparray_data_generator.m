
clear all;
clc;
close all;




out_dir = './output/';
mkdir(out_dir);
fname = textread('list_train_HCC1143_organization.txt','%s');

chn = 4;

for p=1:length(fname)
   
    I_raw = readNPY(sprintf('%s', fname{p}));
        
   
    I_adjust = uint8(zeros(size(I_raw,1),size(I_raw,2),chns));
    for z=1:chns
        I_adjust(:,:,z) = imadjust(uint8(I_raw(:,:,z)*2^8/2^16-1), [0 0.1], []);       
    end
   
  %  subplot(4,4,p)

    I_4ch = uint8(zeros(size(I_raw,1),size(I_raw,2),3));
%     I_4ch(:,:,2) = I_4ch(:,:,2)+I_adjust(:,:,1);
%     I_4ch(:,:,3) = I_4ch(:,:,3)+I_adjust(:,:,1);
%     I_4ch(:,:,1) = I_4ch(:,:,1)+I_adjust(:,:,2);
%     I_4ch(:,:,2) = I_4ch(:,:,2)+I_adjust(:,:,2);
%     I_4ch(:,:,1) = I_4ch(:,:,1)+I_adjust(:,:,3);
%     I_4ch(:,:,3) = I_4ch(:,:,3)+I_adjust(:,:,3);
%     I_4ch(:,:,2) = I_4ch(:,:,2)+I_adjust(:,:,4);
%     
%     I_4ch(:,:,1) = I_4ch(:,:,1)/2;
%     I_4ch(:,:,2) = I_4ch(:,:,2)/3;
%     I_4ch(:,:,3) = I_4ch(:,:,3)/2;

    I_4ch(:,:,3) = I_adjust(:,:,1);
    I_4ch(:,:,2) = I_adjust(:,:,2);
    I_4ch(:,:,1) = I_adjust(:,:,4);
    I_4ch(:,:,2) = I_adjust(:,:,2) + I_adjust(:,:,3);
    I_4ch(:,:,3) = I_adjust(:,:,3) + I_adjust(:,:,3);
    
    I_4ch(:,:,1) = I_4ch(:,:,1)/2;
    I_4ch(:,:,2) = I_4ch(:,:,2)/2;
    
    id = strfind(fname{p}, '/');
    imwrite(I_4ch, sprintf('%s%s.png', out_dir, fname{p}(id(end)+1:end-4)), 'png');

    
end
