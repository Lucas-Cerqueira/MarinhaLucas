function sim_result = sim_nn(nn, input)
	nn.testing = 1;
	net_test = nnff(nn, input, zeros(size(input,1), nn.size(end)));
	nn.testing = 0;

	sim_result = net_test.a{end};
end
