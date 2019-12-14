[train_users, test_users] = loadData('../data/citeulike-a/cf-train-1-users.dat', '../data/citeulike-a/cf-test-1-users.dat');
%load('../model/pmf.mat');
load('../model/pmf_cdl.mat');
recall = evaluatePaper(train_users, test_users, m_U, m_V, 300)
