import numpy as np
import pickle
import torch
from torch.nn import Module	


class SMPL(Module):
	def __init__(self, device=None, model_path='./neutral_smpl_with_cocoplus_reg.pkl', joint_type='cocoplus'):
		super(SMPL, self).__init__()

		with open(model_path, 'rb') as f:
			try:
				params = pickle.load(f)
			except UnicodeDecodeError:  # python 3.x
				f.seek(0)
				params = pickle.load(f, encoding='latin1')
			f.close()

		print(params.keys())

		self.J_regressor = torch.from_numpy(np.array(params['J_regressor'].todense())).type(torch.float64)
		self.weights = torch.from_numpy(params['weights']).type(torch.float64)
		self.posedirs = torch.from_numpy(params['posedirs']).type(torch.float64)
		self.v_template = torch.from_numpy(params['v_template']).type(torch.float64)
		self.shapedirs = torch.from_numpy(np.array(params['shapedirs'])).type(torch.float64)
		self.kintree_table = params['kintree_table']
		self.faces = params['f']
		self.joint_regressor = torch.from_numpy(np.array(params['cocoplus_regressor'].T.todense())).type(torch.float64)
		if joint_type == 'lsp':
			self.joint_regressor = self.joint_regressor[:14]

		self.device = device if device is not None else torch.device('cpu')
		for name in ['J_regressor', 'weights', 'posedirs', 'v_template', 'shapedirs']:
			_tensor = getattr(self, name)
			print('Tensor {} shape {}'.format(name, _tensor.shape))
			setattr(self, name, _tensor.to(self.device))

	@staticmethod
	def rodrigues(r):
		"""
		Rodrigues' rotation formula that turns axis-angle vector into rotation matrix in a batch-ed manner.

		Parameter:
		----------
		r: axis-angle rotattion vector of shape [batch_size, 1, 3].

		Return:
		-------
		rotation maxrix of shape [batch_size, 1, 3].
		"""
		theta = torch.norm(r, dim=(1, 2), keepdim=True)
		torch.max(theta, theta.new_full((1,), torch.finfo(theta.dtype).tiny), out=theta)
		theta_dim = theta.shape[0]
		r_hat = r / theta
		cos = torch.cos(theta)
		z_stick = torch.zeros(theta_dim, dtype=torch.float64).to(r.device)
		m = torch.stack(
		  (z_stick, -r_hat[:, 0, 2], r_hat[:, 0, 1], r_hat[:, 0, 2], z_stick,
		  -r_hat[:, 0, 0], -r_hat[:, 0, 1], r_hat[:, 0, 0], z_stick), dim=1)
		m = torch.reshape(m, (-1, 3, 3))
		i_cube = (torch.eye(3, dtype=torch.float64).unsqueeze(dim=0) \
				 + torch.zeros((theta_dim, 3, 3), dtype=torch.float64)).to(r.device)
		A = r_hat.permute(0, 2, 1)
		dot = torch.matmul(A, r_hat)
		R = cos * i_cube + (1 - cos) * dot + torch.sin(theta) * m
		return R

	@staticmethod
	def with_zeros(x):
		"""
		Append a [0, 0, 0, 1] vector to a [3, 4] matrix.

		Parameter:
		----------
		x: maxtrix to be appedned.

		Return:
		-------
		matrix after appending of shape [4, 4]
		"""
		ones = torch.tensor([[[0.0, 0.0, 0.0, 1.0]]], dtype=torch.float64).expand(x.shape[0], -1, -1).to(x.device)
		ret = torch.cat((x, ones), dim=1)
		return ret

	@staticmethod
	def pack(x):
		"""
		Append zero matrices of shape [4, 3] to vectors [4, 1] shape in a batch-ed manner.

		Parameter:
		----------
		x: Matrices to be appedned of shape [batch_size, 4, 1]

		Return:
		-------
		matrix of shape [batch_size, 4, 4] after appending.
		"""
		zeros43 = torch.zeros((x.shape[0], x.shape[1], 4, 3), dtype=torch.float64).to(x.device)
		rec = torch.cat((zeros43, x), dim=3)
		return rec

	def write_obj(self, verts, file_name):
		with open(file_name, 'w') as fp:
			for v in verts:
				fp.write('v %f %f %f\n' % (v[0], v[1], v[2]))

			for f in self.faces + 1:
				fp.write('f %d %d %d\n' % (f[0], f[1], f[2]))

	def forward(self, beta, theta, trans):
		"""
		Construct a compute graph that takes in parametrs and outputs a tensor as model vertices.

		Parameters:
		----------
		theta: a [N, 24, 3] tensor indicating child joint rotation relative to parent joint.
		beta: a tensor of shape [N, 10] indicating the coefficients for PCA components.
		trans: global translaton of shape [N, 3].

		Return:
		------
		a 3-D tensor of shape [N, 6890, 3] for vercies and the corresponding [N, 19, 3] joint positions.
		"""	
		batch_size = beta.shape[0]
		id_to_col = {
			self.kintree_table[1, i]: i for i in range(self.kintree_table.shape[1])
		}
		parent = {
			i: id_to_col[self.kintree_table[0, i]]
			for i in range(1, self.kintree_table.shape[1])
		}

		# body vertices of shape [N, 3]: T_shaped = T_bar + B_S(beta), where B_S is a blend shape function
		v_shaped = torch.tensordot(beta, self.shapedirs, dims=([1], [2])) + self.v_template
		# K joints location of shape [K, 3]: J = J(beta)
		J = torch.matmul(self.J_regressor, v_shaped)
		
		# rotation matrix for each joint
		R = self.rodrigues(theta.view(-1, 1, 3)).reshape(batch_size, -1, 3, 3)
		R_cube = R[:, 1:, :, :]
		I_cube = (torch.eye(3, dtype=torch.float64).unsqueeze(dim=0) +\
		 torch.zeros((batch_size, R_cube.shape[1], 3, 3), dtype=torch.float64)).to(self.device)
		lrotmin = (R_cube - I_cube).reshape(batch_size, -1, 1).squeeze(dim=2)
		# posed blend shape
		v_posed = v_shaped + torch.tensordot(lrotmin, self.posedirs, dims=([1], [2]))

		# world transformation for each joint
		G = []
		G.append(
			self.with_zeros(torch.cat((R[:, 0], torch.reshape(J[:, 0, :], (-1, 3, 1))), dim=2))
		)
		for i in range(1, self.kintree_table.shape[1]):
			G.append(
				torch.matmul(
					G[parent[i]],
					self.with_zeros(torch.cat((R[:, i], torch.reshape(J[:, i, :]-J[:, parent[i], :], (-1, 3, 1))), dim=2))
				)
			)
		G_stacked = torch.stack(G, dim=1)
		G = G_stacked - \
			self.pack(
				torch.matmul(
					G_stacked,
					torch.reshape(
						torch.cat((J, torch.zeros((batch_size, 24, 1), dtype=torch.float64).to(self.device)), dim=2),
						(batch_size, 24, 4, 1)
					)
				)
			)


		T = torch.tensordot(G, self.weights, dims=([1], [1])).permute(0, 3, 1, 2)
		rest_shape_h = torch.cat(
			(v_posed, torch.ones((batch_size, v_posed.shape[1], 1), dtype=torch.float64).to(self.device)), dim=2
		)
		v = torch.matmul(T, torch.reshape(rest_shape_h, (batch_size, -1, 4, 1)))
		v = torch.reshape(v, (batch_size, -1, 4))[:, :, :3]
		verts = v + torch.reshape(trans, (batch_size, 1, 3))

		# joints
		joints = torch.tensordot(verts, self.joint_regressor, dims=([1], [0])).transpose(1, 2)

		return verts, joints


if __name__ == '__main__':

	device = torch.device('cpu')
	theta_size = 72
	beta_size = 10

	np.random.seed(9608)
	smpl = SMPL(device=device, model_path='neutral_smpl_with_cocoplus_reg.pkl')
	# for i in range(10):
	theta = torch.from_numpy((np.random.rand(10, theta_size) - 0.5) * 0.4).type(torch.float64).to(device)
	beta = torch.from_numpy((np.random.rand(10, beta_size) - 0.5) * 0.06).type(torch.float64).to(device)
	trans = torch.from_numpy(np.zeros((10, 3))).type(torch.float64).to(device)
	verts, joints = smpl(beta, theta, trans)

	print("verts shape: {}".format(verts.shape))
	print("joints shape: {}".format(joints.shape))

	verts_numpy = verts.data.numpy()
	verts = verts_numpy[0]
	print("verst: {}".format(verts[:10]))

	joints_numpy = joints.data.numpy()
	joints = joints_numpy[0]
	print("joint: {}".format(joints[:10]))
	smpl.write_obj(verts, './smpl_torch.obj')