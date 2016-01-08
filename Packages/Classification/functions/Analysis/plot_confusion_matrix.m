function [h] = plot_confusion_matrix(target,output,threshold)
%PLOT_CONFUSION_MATRIX Summary of this function goes here
%   Detailed explanation goes here

h = figure;

if length(target) ~= length(output)
    error('plot_confusion_matrix: Target and Output with different sizes');
end

classification_target = target;
classification_output = (output > threshold);

[~,conf_mat,~,~] = confusion(classification_target,classification_output);

confusion_matrix = zeros(size(conf_mat));

for iclass = 1:size(conf_mat,1)
    for jclass = 1:size(conf_mat,2)
        confusion_matrix(iclass,jclass) = 100*(conf_mat(iclass,jclass)/sum(sum(conf_mat(iclass,:))));
    end
end

imagesc(confusion_matrix);
colormap(flipud(gray)); % gray map
colorbar;

textStrings = num2str(confusion_matrix(:),'%1.2f%%');
textStrings = strtrim(cellstr(textStrings));
[x,y] = meshgrid(1:length(confusion_matrix));
hStrings = text(x(:),y(:),textStrings(:),'HorizontalAlignment','center','FontWeight','bold'); % plot strings

midValue = mean(get(gca,'CLim'));  % Get the middle value of the color range
textColors = repmat(confusion_matrix(:) > midValue,1,3);  % Choose white or black for the
                                             %   text color of the strings so
                                             %   they can be easily seen over
                                             %   the background color
                                             
set(hStrings,{'Color'},num2cell(textColors,2));  % Change the text colors

title(sprintf('Confusion Matrix'),'FontSize',20,'FontWeight','bold');
xlabel('Actual','FontSize', 15,'FontWeight', 'bold');
ylabel('Predicted','FontSize', 15,'FontWeight', 'bold');
        
l_aux = {'ClassA','ClassB','ClassC','ClassD'};
set(gca,'XTick',1:4);
set(gca,'YTick',1:4);
set(gca,'XTickLabel',l_aux,'fontWeight','bold');
set(gca,'YTickLabel',l_aux,'fontWeight','bold');


end

