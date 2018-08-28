Files=dir('*.bin');
for k=1:length(Files)
   FileNames = Files(k).name;
   fileID = fopen(FileNames,'r');
   mask=fread(fileID);
   mask=reshape(mask,256,256);
   Mask=(mask==1)';
   save(strcat(FileNames(1:end-4),'.mat'),'Mask');
   fclose(fileID);
end