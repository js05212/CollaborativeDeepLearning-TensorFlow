function [train_users, test_users] = loadData(ftrain, ftest)

l_train = textread(ftrain, '%s', 'delimiter', '\n');
num_train = size(l_train, 1);
train_users = cell(num_train, 1);
for i = 1:num_train
    train_users{i} = str2num(l_train{i});
    %train_users{i}(2:end) = train_users{i}(2:end) + 1;
end

l_test = textread(ftest, '%s', 'delimiter', '\n');
num_test = size(l_test, 1);
test_users = cell(num_test, 1);
for i = 1:num_test
    test_users{i} = str2num(l_test{i}) + 1;
    %test_users{i}(2:end) = test_users{i}(2:end) + 1;
end
