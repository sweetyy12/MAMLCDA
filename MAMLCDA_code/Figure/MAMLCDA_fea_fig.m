%get figure
data=csvread('fea.csv',1,1);  
rowshu=size(data,1);
%print(rowshu)
for i = 1:rowshu
    x=1:1:421; 
    y=data(i,:);
    hold on
    plot(x,y);
    hold off
    frame = getframe; 
    im = frame.cdata; 
    str=['D:\feafig\','k_row',num2str(i),'_yes.jpg']; 
    imwrite(im,str); 
    close(gcf);
end
