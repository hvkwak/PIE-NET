function Vis_closed_final()

FindFiles_test_data =  '.\test_result_2_1\'; 
Files = dir(fullfile(FindFiles_test_data));
filenames = {Files.name}';
filenames = filenames(3:length(filenames));
filenames=filenames';

load color

 for i = 1:length(filenames)
    load([FindFiles_test_data,filenames{i}]);
    [num_data, ~,~] = size(input_points_edge_corner);
    for j =1:num_data
        tic
       %% input points
        input_points = squeeze(input_points_edge_corner(j,:,1:3));
       %%  figure1: prediction open_and_closed edge points
        open_edge_points_pre = squeeze(pre_label_5(j,:,:));
        open_edge_points_pre = exp(open_edge_points_pre);
        sum_open_edge_pre = sum(open_edge_points_pre,2);
        open_edge_points_pre = open_edge_points_pre./repmat(sum_open_edge_pre,1,2);
        open_edge_points_pre = find(open_edge_points_pre(:,2)>0.5);

        closed_edge_points_pre = squeeze(pre_label_6(j,:,:));
        closed_edge_points_pre = exp(closed_edge_points_pre);
        sum_closed_edge_pre = sum(closed_edge_points_pre,2);
        closed_edge_points_pre = closed_edge_points_pre./repmat(sum_closed_edge_pre,1,2);
        closed_edge_points_pre = find(closed_edge_points_pre(:,2)>0.5);
        closed_edge_points_pre = setdiff(closed_edge_points_pre,open_edge_points_pre);
        
        All_open_closed_edge_points_pre = [open_edge_points_pre;closed_edge_points_pre];
        All_open_closed_edge_points_pre_label = [ones(1,size(open_edge_points_pre,1)),zeros(1,size(closed_edge_points_pre,1))];
        All_open_closed_edge_points = input_points(All_open_closed_edge_points_pre,:);
        closed_edge_points = input_points(closed_edge_points_pre,:);
        dis_closed_all = Distance_Points1_Points2_matrix(closed_edge_points , All_open_closed_edge_points);
        num_neibor_from_all = 6;
        [~,idx_neibor_from_all] = sort(dis_closed_all,2);
        idx_neibor_from_all = idx_neibor_from_all(:,2:num_neibor_from_all);
        closed_edge_points_ind = zeros(size(closed_edge_points,1),num_neibor_from_all-1);
        for i_closed_edge = 1:size(closed_edge_points,1)
           closed_edge_points_ind(i_closed_edge,:) = All_open_closed_edge_points_pre_label(idx_neibor_from_all(i_closed_edge,:));
        end
        closed_edge_points_ind_sum =  sum(closed_edge_points_ind,2);
        true_closed_edge_idx = find(closed_edge_points_ind_sum<num_neibor_from_all/2);
        true_closed_edge_points_pre = closed_edge_points_pre(true_closed_edge_idx);
        
        [global_closed_edge_idx_unique,~]= unique(true_closed_edge_points_pre);
        if isempty(global_closed_edge_idx_unique)
            continue
        end
        

        fig_1 = figure(1);
        open_edge_points = input_points(open_edge_points_pre,:);
        temp_points = open_edge_points;
        temp_color = color(1,:);  
        temp_color = repmat(temp_color,size(open_edge_points,1),1);
        temp_all = [temp_points,temp_color];
        scatter3(temp_all(:,1),temp_all(:,2),temp_all(:,3),50,temp_all(:,4:6),'.'); 
        hold on
        
        closed_edge_points = input_points(true_closed_edge_points_pre,:);
        temp_points = closed_edge_points;
        temp_color = color(2,:);  
        temp_color = repmat(temp_color,size(closed_edge_points,1),1);
        temp_all = [temp_points,temp_color];
        scatter3(temp_all(:,1),temp_all(:,2),temp_all(:,3),50,temp_all(:,4:6),'.');              
        
        axis equal
        title('Predicted open and closed points')       
        hold off

           
       %%         
        closed_edge_points = input_points(global_closed_edge_idx_unique,:);       
        closed_sample_all = size(closed_edge_points,1);
        if size(closed_edge_points,1)<20
            continue;
        end
        
        num_neibor = 4;
        Weight_matrix = Distance_Points1_Points2_matrix(closed_edge_points, closed_edge_points);        
        [val_test,idx_wei_mat_1] = sort(Weight_matrix,2);
        idx_wei_mat = idx_wei_mat_1(:,2:num_neibor);
        Weight_matrix_mask = zeros(size(closed_edge_points,1),size(closed_edge_points,1));
        for row_wei_mat = 1:closed_sample_all
            temp_mask = zeros(1,closed_sample_all);
            temp_mask(idx_wei_mat(row_wei_mat,:)) = 1;
            Weight_matrix_mask(row_wei_mat,:) = temp_mask;
        end
        Weight_matrix = Weight_matrix.* Weight_matrix_mask;
        Weight_matrix = Weight_matrix + Weight_matrix';
        
        G = graph(Weight_matrix);
        G_edges = table2array(G.Edges);
        
        target = [1:closed_sample_all];
        
        if closed_sample_all>500
            [~,down_sample_point_idx] = Farthest_Point_Sampling_piont_and_idx(closed_edge_points,500); % local idx
            source = down_sample_point_idx;
        else
            source = [1:closed_sample_all];
        end
        
        closed_sample = numel(source);
        path_cell_all = cell(closed_sample,1);
        path_point_idx_cell_all = cell(closed_sample,1);
        for i_source = 1:closed_sample
          [TR,D] = shortestpathtree(G,source(i_source),target,'OutputForm','cell');
          path_length = cellfun(@(x) size(x,2), TR, 'Unif',0);
          path_length = cell2mat(path_length);
          
          path_idx_idx = find(path_length>= 6 & path_length<=40);
          path_cell = TR(path_idx_idx);
          
          if isempty(path_cell)
              path_point_idx_mat = [];
          else
              path_point_idx_cell = cellfun(@(x) Find_path_idx(x), path_cell, 'Unif',0);
              path_point_idx_mat = cell2mat(path_point_idx_cell);
          end
          path_point_idx_cell_all{i_source} = path_point_idx_mat;
          
          path_cell_all{i_source} = path_cell;
        end
 
        
       %%  
        All_proposals_sample_points = cell(closed_sample,1);
        All_proposals_three_points = cell(closed_sample,1);
        All_proposals_center_points = cell(closed_sample,1); 
        All_proposals_rad = cell(closed_sample,1); 
        for c = 1:closed_sample
          temp_path_idx_mat = path_point_idx_cell_all{c};
          num_path = size(temp_path_idx_mat,1);
          
          temp_sample_points_cell = {};
          temp_three_points_cell = {};
          temp_center_points_cell = {};
          temp_rad_cell = {};
          for path_i = 1:num_path
              P1 = closed_edge_points(temp_path_idx_mat(path_i,1),:);
              P2 = closed_edge_points(temp_path_idx_mat(path_i,2),:);
              P3 = closed_edge_points(temp_path_idx_mat(path_i,3),:);
              [three_points_cell, sample_points_cell, center_points_cell, rad_cell] = generate_three_candidates(P1,P2,P3,64);
              temp_sample_points_cell = [temp_sample_points_cell; sample_points_cell];
              temp_three_points_cell = [temp_three_points_cell; three_points_cell];
              temp_center_points_cell = [temp_center_points_cell; center_points_cell];
              temp_rad_cell = [temp_rad_cell; rad_cell];
          end
              [idx_path] = Compute_match_search_one(temp_sample_points_cell,closed_edge_points,path_cell_all{c});
              if isempty(idx_path)
                  All_proposals_sample_points{c} = [];
                  All_proposals_three_points{c} = [];
                  All_proposals_center_points{c} = [];
                  All_proposals_rad{c} = [];
              else
                  All_proposals_sample_points{c} = temp_sample_points_cell{idx_path};
                  All_proposals_three_points{c} = temp_three_points_cell{idx_path};
                  All_proposals_center_points{c} = temp_center_points_cell{idx_path}; 
                  All_proposals_rad{c} = temp_rad_cell{idx_path};
              end
        end
        
       %% pre_center < (min_x,min_y,min_z) or pre_center > (max_x,max_y,max_z)
        % aabb bounding box
        min_xyz = min(closed_edge_points);
        max_xyz = max(closed_edge_points);        
        min_max_vector = min_xyz - max_xyz;        
        max_min_vector = max_xyz - min_xyz;
        min_xyz = min_xyz + 0.15*min_max_vector;
        max_xyz = max_xyz + 0.15*max_min_vector;
                
        % delete [] cell 
        All_proposals_sample_points(cellfun(@isempty,All_proposals_sample_points))=[]; 
        All_proposals_center_points(cellfun(@isempty,All_proposals_center_points))=[];
        All_proposals_rad(cellfun(@isempty,All_proposals_rad))=[];
        
        % delete proposal beyond aabb 
        aabb = [min_xyz; max_xyz];
        All_proposals_sample_points_blag = cellfun(@(x) Judge_aabb_pointset(aabb, x), All_proposals_sample_points,'Unif',0);
        All_proposals_sample_points_blag = cell2mat(All_proposals_sample_points_blag);
        proposal_idx = find(All_proposals_sample_points_blag==1);
        All_proposals_sample_points_range = All_proposals_sample_points(proposal_idx);
        All_proposals_center_points_range = All_proposals_center_points(proposal_idx);
        All_proposals_rad_range = All_proposals_rad(proposal_idx);     

    
 
       %% fig 3: all prediction closed points and  All closed curves
        fig_2 = figure(2);
        % visulization predict open and closed edge points and all predicted cycles;
        open_edge_points = input_points(open_edge_points_pre,:);
        temp_points = open_edge_points;
        temp_color = color(1,:);  
        temp_color = repmat(temp_color,size(open_edge_points,1),1);
        temp_all = [temp_points,temp_color];
        scatter3(temp_all(:,1),temp_all(:,2),temp_all(:,3),50,temp_all(:,4:6),'.'); 
        hold on
        
        closed_edge_points = input_points(true_closed_edge_points_pre,:);
        temp_points = closed_edge_points;
        temp_color = color(2,:);  
        temp_color = repmat(temp_color,size(closed_edge_points,1),1);
        temp_all = [temp_points,temp_color];
        scatter3(temp_all(:,1),temp_all(:,2),temp_all(:,3),50,temp_all(:,4:6),'.');
        hold on       
        axis equal
        
        %all pred cycles;
        num_proposal_range = numel(All_proposals_sample_points_range);
        for x = 1:num_proposal_range
            pre_curve_points = All_proposals_sample_points_range{x};
            pre_curve_points_color = color(3,:); 
            num_points_pre_curve = size(pre_curve_points,1);
            pre_curve_points_color = repmat(pre_curve_points_color,num_points_pre_curve,1);
            pre_curve_all = [pre_curve_points,pre_curve_points_color];
            scatter3(pre_curve_all(:,1),pre_curve_all(:,2),pre_curve_all(:,3),50,pre_curve_all(:,4:6),'.');   
            hold on            
        end

        axis equal
        title('All Predicted Cycles');
        hold off       
        
    %% 
     pre_cyc_cyc_pick = merge_all_pre_cycle(All_proposals_rad_range, closed_edge_points); %       
     vis_cyc(pre_cyc_cyc_pick,input_points,open_edge_points_pre,true_closed_edge_points_pre);
     
    disp('i = ')
    disp(i)
    disp('j = ')
    disp(j)        
    toc
    end  
 end
end 

function vis_cyc(all_cyc_cell,input_points,open_edge_points_pre,true_closed_edge_points_pre)
        load color
        fig_3 = figure(3);

        open_edge_points = input_points(open_edge_points_pre,:);
        temp_points = open_edge_points;
        temp_color = color(1,:);  
        temp_color = repmat(temp_color,size(open_edge_points,1),1);
        temp_all = [temp_points,temp_color];
        scatter3(temp_all(:,1),temp_all(:,2),temp_all(:,3),50,temp_all(:,4:6),'.'); 
        hold on
        
        closed_edge_points = input_points(true_closed_edge_points_pre,:);
        temp_points = closed_edge_points;
        temp_color = color(2,:);  
        temp_color = repmat(temp_color,size(closed_edge_points,1),1);
        temp_all = [temp_points,temp_color];
        scatter3(temp_all(:,1),temp_all(:,2),temp_all(:,3),50,temp_all(:,4:6),'.'); 
        hold on
        
        
        num_cyc = numel(all_cyc_cell);
        for x = 1:num_cyc
            pre_curve_points = all_cyc_cell{x};
            pre_curve_points_color = color(3,:); 
            num_points_pre_curve = size(pre_curve_points,1);
            pre_curve_points_color = repmat(pre_curve_points_color,num_points_pre_curve,1);
            pre_curve_all = [pre_curve_points,pre_curve_points_color];
            scatter3(pre_curve_all(:,1),pre_curve_all(:,2),pre_curve_all(:,3),50,pre_curve_all(:,4:6),'.');   
            hold on
            axis equal
        end

        axis equal
        title('Final result');
        hold off
end


function pre_cyc_cyc_pick = merge_all_pre_cycle(pre_cyc_para_cell, target_pointset)
    res_thr = 0.5; 
    overlap_thr = 0.3;    

   if isempty(pre_cyc_para_cell)
      pre_cyc_cyc_pick =  [];
      return;
   end

   %
   pre_cyc_points_cell = cellfun(@(x) quick_render(x,360),pre_cyc_para_cell,'Unif',0);
   
   %
   num_pre_cyc = numel(pre_cyc_para_cell);
   [res_cell,idx_cell] = compute_project_residual(pre_cyc_points_cell,target_pointset);
   pre_cyc_res_mat = cell2mat(res_cell);
   pre_cyc_idx_mat = cell2mat(idx_cell);

   
   %
   pre_cyc_para_mat = cell2mat(pre_cyc_para_cell);
   pre_cyc_para_rad = pre_cyc_para_mat(:,1);
   pre_cyc_para_scale = 1./pre_cyc_para_rad; 
   pre_cyc_res_mat = pre_cyc_res_mat.*pre_cyc_para_scale;
   
   %
   
   idx_res = find(pre_cyc_res_mat < res_thr);
   pre_cyc_res_mat = pre_cyc_res_mat(idx_res,:);
   pre_cyc_idx_mat = pre_cyc_idx_mat(idx_res,:);
   pre_cyc_points_cell = pre_cyc_points_cell(idx_res,:);
   pre_cyc_para_scale = pre_cyc_para_scale(idx_res,:);
   
   if isempty(pre_cyc_points_cell)
      pre_cyc_cyc_pick =  [];
      return;
   end
   
   %     
   pick_idx = NMS_PIE(pre_cyc_idx_mat, pre_cyc_res_mat, overlap_thr);
   
   %
   pre_cyc_cyc_pick = pre_cyc_points_cell(pick_idx);
   
end


function [res_cell,idx_cell] = compute_project_residual(Points_1_cell,Points_2)

num_cell = numel(Points_1_cell);
res_cell = cell(num_cell,1);
idx_cell = cell(num_cell,1);
for i = 1:num_cell
   Dis_matrix = Distance_Points1_Points2(Points_1_cell{i},Points_2);
   [dist_order_mat,idx_order_mat] = sort(Dis_matrix,2);
   dist_vector = sqrt(dist_order_mat(:,1));
   idx_vector = idx_order_mat(:,1);
   idx_vector = idx_vector';
   res = mean(dist_vector); % max(dist_vector);
   res_cell{i} = res;
   idx_cell{i} = idx_vector;
end

end

function  path_th = Find_path_idx(path_idx)
    num_eles = numel(path_idx);
    nei_1_idx = round(num_eles/2);
    path_th(1) = path_idx(1);
    path_th(2) = path_idx(nei_1_idx);
    path_th(3) = path_idx(end);
end

function [idx] = Compute_match_search_one(proposals_cell,all_feature_points,path_cell)
   % all element in path_cell
   num_path_cell = numel(path_cell);
   all_path_idx = [];
   for i = 1:num_path_cell
       all_path_idx = [all_path_idx, path_cell{i}];
   end
   
  all_path_idx = unique(all_path_idx);
   
  num_all_path_idx = numel(all_path_idx);
  if num_all_path_idx > 500 
      [down_sample_points,~] = Farthest_Point_Sampling_piont_and_idx(all_feature_points,500); % local idx
  else
      down_sample_points = all_feature_points(all_path_idx,:);
  end
   
   num_pro = numel(proposals_cell);
   score = zeros(num_pro,1);
   for i = 1:num_pro
      temp_proposal = proposals_cell{i};
      if isempty(temp_proposal)
          dist = 10000;
      else
          %[temp_proposal,~] = Farthest_Point_Sampling_piont_and_idx(temp_proposal,16);
          [dist] = hausdorff(temp_proposal, down_sample_points); 
      end
      score(i) = dist;
   end   
   [val,idx] = min(score);
end

function [All_proposals_three_points, All_proposals_sample_points, All_proposals_center_points,All_proposals_rad] = generate_three_candidates(P1,P2,P3,num_points)
          
          temp_points = [P1;P2;P3];          
          temp_points_x = temp_points;
          temp_points_x(2,1) = temp_points_x(1,1);
          temp_points_x(3,1) = temp_points_x(1,1);
          temp_points_x_2 = temp_points_x(:,[2,3]);
          P1 = temp_points_x_2(1,:);
          P2 = temp_points_x_2(2,:);
          P3 = temp_points_x_2(3,:);
          [Result, cc_x, rad_x] = ThreePoint2Circle(P1, P2, P3, num_points);
          if isempty(Result)
              cirxyz_x = [];
              center_x = [];
              rad_center_poi_x = [];
          else
              cirxyz_x = [repmat(temp_points_x(1,1),size(Result,1),1), Result(:,1), Result(:,2)];
              center_x = [temp_points_x(1,1),cc_x(1),cc_x(2)];
              rad_center_poi_x = [rad_x,cc_x(1),cc_x(2),1,temp_points_x(1,1)]; 
          end
          All_proposals_three_points{1,1} = temp_points_x;
          All_proposals_sample_points{1,1} = cirxyz_x;
          All_proposals_center_points{1,1} = center_x;
          All_proposals_rad{1,1} = rad_center_poi_x;         
          
          temp_points_y = temp_points;
          temp_points_y(2,2) = temp_points_y(1,2);
          temp_points_y(3,2) = temp_points_y(1,2);   

          
          temp_points_y_2 = temp_points_y(:,[1,3]);
          P1 = temp_points_y_2(1,:);
          P2 = temp_points_y_2(2,:);
          P3 = temp_points_y_2(3,:);
          [Result,cc_y,rad_y] = ThreePoint2Circle(P1, P2, P3, num_points);
          if isempty(Result)
              cirxyz_y = [];
              center_y = [];
              rad_center_poi_y = [];
          else
              cirxyz_y = [Result(:,1),repmat(temp_points_y(1,2),size(Result,1),1), Result(:,2)];
              center_y = [cc_y(1),temp_points_y(1,2),cc_y(2)];
              rad_center_poi_y = [rad_y, cc_y(1),cc_y(2),2,temp_points_y(1,2)];
          end
          
          
          All_proposals_three_points{2,1} = temp_points_y;
          All_proposals_sample_points{2,1} = cirxyz_y;
          All_proposals_center_points{2,1} = center_y;
          All_proposals_rad{2,1} = rad_center_poi_y; 
          
          temp_points_z = temp_points;
          temp_points_z(2,3) = temp_points_z(1,3);
          temp_points_z(3,3) = temp_points_z(1,3);
          temp_points_z_2 = temp_points_z(:,[1,2]);
          P1 = temp_points_z_2(1,:);
          P2 = temp_points_z_2(2,:);
          P3 = temp_points_z_2(3,:);
          [Result,cc_z, rad_z] = ThreePoint2Circle(P1, P2, P3, num_points);
          
          if isempty(Result)
              cirxyz_z = [];
              center_z = [];
              rad_center_poi_z = [];
          else
              cirxyz_z = [Result(:,1), Result(:,2),repmat(temp_points_z(1,3),size(Result,1),1)];
              center_z = [cc_z(1),cc_z(2),temp_points_z(1,3)]; 
              rad_center_poi_z = [rad_z,cc_z(1),cc_z(2),3,temp_points_z(1,3)];
          end
          All_proposals_three_points{3,1} = temp_points_z;
          All_proposals_sample_points{3,1} = cirxyz_z;
          All_proposals_center_points{3,1} = center_z; 
          All_proposals_rad{3,1} = rad_center_poi_z;

end

function Del_blag = Judge_aabb_pointset(aabb, pointset)
   % aabb bounding box
   min_xyz = min(pointset);
   max_xyz = max(pointset);
   
   min_aabb = aabb(1,:);
   max_aabb = aabb(2,:);
   
   if min_xyz(1)<min_aabb(1)||min_xyz(2)<min_aabb(2)||min_xyz(3)<min_aabb(3)||max_xyz(1)>max_aabb(1)||max_xyz(2)>max_aabb(2)||max_xyz(3)>max_aabb(3)
       Del_blag = 0;
   else 
       Del_blag = 1;
   end

end



function Dis_matrix = Distance_Points1_Points2_matrix(vertices_ball,vertices_points)

B=vertices_ball; 
P=vertices_points; 
B1=sum(B.^2,2);
P1=sum(P.^2,2);
Num_b=numel(B1);
Num_p=numel(P1);
Dis_matrix= repmat(B1,1,Num_p)+ repmat(P1',Num_b,1) - 2*B*P';  
Dis_matrix(Dis_matrix<0) = 0;

end


function Dis_matrix = Distance_Points1_Points2(vertices_ball,vertices_points)

B=vertices_ball; 
P=vertices_points; 
B1=sum(B.^2,2);
P1=sum(P.^2,2);
Num_b=numel(B1);
Num_p=numel(P1);
Dis_matrix= repmat(B1,1,Num_p)+ repmat(P1',Num_b,1) - 2*B*P';  
end
 

function [Result, P0, R] = ThreePoint2Circle(P1, P2, P3, num)

warning('off');
x1 = P1(1);    x2 = P2(1);    x3 = P3(1);
y1 = P1(2);    y2 = P2(2);    y3 = P3(2);

    if area(P1,P2,P3)<=1e-5
        Result = [];
        P0 = [];
        R = [];
    else
        z1 = x2^2 + y2^2 - x1^2 - y1^2;
        z2 = x3^2 + y3^2 - x1^2 - y1^2;
        z3 = x3^2 + y3^2 - x2^2 - y2^2;
        A = [(x2-x1), (y2-y1); (x3-x1), (y3-y1); (x3-x2), (y3-y2)];
        B = 0.5*[z1;  z2;  z3];
        P0 = (A'*A)\A'*B;
        R1 = sqrt( (P0(1) - P1(1))^2 + (P0(2) - P1(2))^2 );
        R2 = sqrt( (P0(1) - P2(1))^2 + (P0(2) - P2(2))^2 );
        R3 = sqrt( (P0(1) - P3(1))^2 + (P0(2) - P3(2))^2 );
        R = (R1 + R2 + R3)/3;

        theta = (0:2*pi/num:2*pi)';
        Result = zeros(size(theta,1),2);
        for i = 1: size(theta,1)
            Result(i,1) = P0(1) + R*cos(theta(i));
            Result(i,2) = P0(2) + R*sin(theta(i));
        end
    end
end

function cirxyz = quick_render(rad,num)
    P0 = rad(2:3);
    R = rad(1);
    theta = (0:2*pi/num:2*pi)';
    Result = zeros(size(theta,1),2);
    for i = 1: size(theta,1)
        Result(i,1) = P0(1) + R*cos(theta(i));
        Result(i,2) = P0(2) + R*sin(theta(i));
    end

    temp_position = rad(4);
    temp_xyz = rad(5);

    if temp_position == 1
        cirxyz = [repmat(temp_xyz,size(Result,1),1),Result(:,1),Result(:,2)];
    elseif temp_position == 2
        cirxyz = [Result(:,1),repmat(temp_xyz,size(Result,1),1),Result(:,2)];
    elseif temp_position == 3
        cirxyz = [Result(:,1),Result(:,2),repmat(temp_xyz,size(Result,1),1)];
    end
end

function S = area(P_1,P_2,P_3)
    A = P_1;
    B = P_2;
    C = P_3;
    a = sqrt((A(1)-B(1))^2+(A(2)-B(2))^2); 
    b = sqrt((C(1)-B(1))^2+(C(2)-B(2))^2); 
    c = sqrt((A(1)-C(1))^2+(A(2)-C(2))^2); 
    p = (a+b+c)/2;
    S = sqrt(p*(p-a)*(p-b)*(p-c));
end 



 
 
 
 
 
 
 
 
        