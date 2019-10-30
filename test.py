import numpy as np
import pickle

with open('neutral_smpl_with_cocoplus_reg.pkl', 'rb') as f:
	params_1 = pickle.load(f, encoding="latin-1")
	f.close()

with open('model.pkl', 'rb') as f:
	params_2 = pickle.load(f)
	f.close()


print(params_1['shapedirs'][0] == params_2['shapedirs'][0])

print(params_1['shapedirs'][0:10,0,0])
print(params_2['shapedirs'][0:10, 0, 0])

