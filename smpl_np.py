import numpy as np
import pickle

class SMPL():
	def __init__(self, model_path):
		"""
		SMPL model:
		N = 6890 vertuces
		K = 23 joints
		beta = 10 shape parameters
		theta = 72 (24 * 3) pose parameters
		"""
		with open(model_path, 'rb') as f:
			try:
				params = pickle.load(f)
			except UnicodeDecodeError:  # python 3.x
				f.seek(0)
				params = pickle.load(f, encoding='latin1')
			f.close()

		print(params.keys())

		# J regression function predict K joint locations given beta: J
		self.J_regressor = params['J_regressor']
		# W weight matrix
		self.weights = params['weights']
		# P pose blend shapes accounts for the effects of pose-dependent deformations [N, 3, 9K]
		self.posedirs = params['posedirs']
		# template mesh vertices: T_bar 
		self.v_template = params['v_template']
		# shape displacement matrix S of shape [N, 3, beta_shape]
		self.shapedirs = params['shapedirs']
		self.faces = params['f']
		self.kintree_table = params['kintree_table']

		id_to_col = {
			self.kintree_table[1, i]: i for i in range(self.kintree_table.shape[1])
		}
		self.parent = {
			i: id_to_col[self.kintree_table[0, i]]
			for i in range(1, self.kintree_table.shape[1])
		}

		# theta contain the pose parameters
		# beta contain the shape parameters
		# trans contain global translation
		self.theta_shape = [24, 3]
		self.beta_shape = [10]
		self.trans_shape = [3]

		self.theta = np.zeros((self.theta_shape))
		self.beta = np.zeros((self.beta_shape))
		self.trans = np.zeros((self.trans_shape))

		self.verts = None
		self.J = None
		self.R = None

		self.update()

	def set_params(self, theta=None, beta=None, trans=None):
		"""
		Set pose (theta), shape (beta) and translation parameters of SMPL model.
		Vertices of the model will be updated and returned.

		Parameters:
		----------
		theta: a [24, 3] matrix indicating child joint rotation relative to parent joint.
		beta: a vector of shape [10] indicating the coefficients for PCA component.
		trans: global translaton of shape [3].

		Return:
		-------
		Updated vertices
		"""

		if theta is not None:
			self.theta = theta
		if beta is not None:
			self.beta = beta
		if trans is not None:
			self.trans = trans
		self.update()
		return self.verts

	def update(self):
		"""
		Called automatically when parameters are uploaded
		"""

		# body vertices of shape [N, 3]: T_shaped = T_bar + B_S(beta), where B_S is a blend shape function
		v_shaped = self.shapedirs.dot(self.beta) + self.v_template
		# K joints location of shape [K, 3]: J = J(beta)
		self.J = self.J_regressor.dot(v_shaped)

		theta_cube = self.theta.reshape((-1, 1, 3))
		# rotation matrix for each joint
		self.R = self.rodrigues(theta_cube)
		I_cude = np.broadcast_to(
			np.expand_dims(np.eye(3), axis=0),
			(self.R.shape[0]-1, 3, 3)
		)
		lrotmin = (self.R[1:] - I_cude).ravel()
		# posed blend shape: 
		v_posed = v_shaped + self.posedirs.dot(lrotmin)

		# world transformation of each joint
		G = np.empty((self.kintree_table.shape[1], 4, 4))
		G[0] = self.with_zeros(np.hstack((self.R[0], self.J[0, :].reshape([3, 1]))))
		for i in range(1, self.kintree_table.shape[1]):
			G[i] = G[self.parent[i]].dot(
			self.with_zeros(
			  np.hstack(
				[self.R[i],((self.J[i, :]-self.J[self.parent[i],:]).reshape([3,1]))]
			  )
			)
		)
		# remove the transformation due to the rest pose
		G = G - self.pack(
		  np.matmul(
			G,
			np.hstack([self.J, np.zeros([24, 1])]).reshape([24, 4, 1])
		  )
		)

		# transformation of each vertex
		T = np.tensordot(self.weights, G, axes=[[1], [0]])
		rest_shape_h = np.hstack((v_posed, np.ones([v_posed.shape[0], 1])))
		v = np.matmul(T, rest_shape_h.reshape([-1, 4, 1])).reshape([-1, 4])[:, :3]
		self.verts = v + self.trans.reshape([1, 3])


	def rodrigues(self, r):
		"""
		Rodrigues' rotation formula that turns axis-angle vector into rotation matrix in a batch-ed manner.

		Parameter:
		----------
		r: axis-angle rotattion vector of shape [batch_size, 1, 3].

		Return:
		-------
		rotation maxrix of shape [batch_size, 1, 3].
		"""

		theta = np.linalg.norm(r, axis=(1, 2), keepdims=True)
		# avoid zero divide
		theta = np.maximum(theta, np.finfo(np.float64).tiny)
		r_hat = r / theta
		cos = np.cos(theta)
		z_stick = np.zeros(theta.shape[0])
		m = np.dstack([
		  z_stick, -r_hat[:, 0, 2], r_hat[:, 0, 1],
		  r_hat[:, 0, 2], z_stick, -r_hat[:, 0, 0],
		  -r_hat[:, 0, 1], r_hat[:, 0, 0], z_stick]
		).reshape([-1, 3, 3])
		i_cube = np.broadcast_to(
		  np.expand_dims(np.eye(3), axis=0),
		  [theta.shape[0], 3, 3]
		)
		A = np.transpose(r_hat, axes=[0, 2, 1])
		B = r_hat
		dot = np.matmul(A, B)
		R = cos * i_cube + (1 - cos) * dot + np.sin(theta) * m
		return R

	def with_zeros(self, x):
		"""
		Append a [0, 0, 0, 1] vector to a [3, 4] matrix.

		Parameter:
		----------
		x: maxtrix to be appedned.

		Return:
		-------
		matrix after appending of shape [4, 4]
		"""
		return np.vstack((x, np.array([0.0, 0.0, 0.0, 1.0])))

	def pack(self, x):
		"""
		Append zero matrices of shape [4, 3] to vectors [4, 1] shape in a batch-ed manner.

		Parameter:
		----------
		x: Matrices to be appedned of shape [batch_size, 4, 1]

		Return:
		-------
		matrix of shape [batch_size, 4, 4] after appending.
		"""

		return np.dstack((np.zeros((x.shape[0], 4, 3)), x))

	def save_to_obj(self, path):
		"""
		Save the SMPL model into .obj file.

		Parameter:
		----------
		path: path to save.

		"""
		with open(path, 'w') as fp:
			for v in self.verts:
				fp.write('v %f %f %f\n' % (v[0], v[1], v[2]))
			for f in self.faces + 1:
				fp.write('f %d %d %d\n' % (f[0], f[1], f[2]))




if __name__ == '__main__':
	smpl = SMPL(model_path='neutral_smpl_with_cocoplus_reg.pkl')
	np.random.seed(9608)
	theta = (np.random.rand(*smpl.theta.shape) - 0.5) * 0.4
	beta = (np.random.rand(*smpl.beta.shape) - 0.5) * 0.06
	trans = np.zeros(smpl.trans.shape)
	smpl.set_params(beta=beta, theta=theta, trans=trans)
	smpl.save_to_obj('./smpl_np.obj')
	print(smpl.verts[:10])
