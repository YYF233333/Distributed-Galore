import numpy as np
import torch
import time
from concurrent.futures import Future
from concurrent.futures import ThreadPoolExecutor

time_acc = 0

thread_pool = ThreadPoolExecutor(max_workers=20)

class GaLoreProjector:

    def __init__(self, rank, verbose=False, update_proj_gap=200, scale=1.0, proj_type='std'):
        self.rank = rank
        self.verbose = verbose
        self.update_proj_gap = update_proj_gap
        self.scale = scale
        self.next_ortho_matrix = None
        self.ortho_matrix = None
        self.proj_type = proj_type
        # 异步相关
        self.last_result = None
        
    def project(self, full_rank_grad, iter):
        if self.proj_type == 'std':
            if full_rank_grad.shape[0] >= full_rank_grad.shape[1]:
                if self.ortho_matrix is None or iter % self.update_proj_gap == 0:
                    self.ortho_matrix = self.get_orthogonal_matrix(full_rank_grad, self.rank, type='right')
                low_rank_grad = torch.matmul(full_rank_grad, self.ortho_matrix.t())
            else:
                if self.ortho_matrix is None or iter % self.update_proj_gap == 0:
                    self.ortho_matrix = self.get_orthogonal_matrix(full_rank_grad, self.rank, type='left')
                low_rank_grad = torch.matmul(self.ortho_matrix.t(), full_rank_grad)
        elif self.proj_type == 'reverse_std':
            if full_rank_grad.shape[0] >= full_rank_grad.shape[1]:
                if self.ortho_matrix is None or iter % self.update_proj_gap == 0:
                    self.ortho_matrix = self.get_orthogonal_matrix(full_rank_grad, self.rank, type='left')
                low_rank_grad = torch.matmul(self.ortho_matrix.t(),full_rank_grad)
            else:
                if self.ortho_matrix is None or iter % self.update_proj_gap == 0:
                    self.ortho_matrix = self.get_orthogonal_matrix(full_rank_grad, self.rank, type='right')
                low_rank_grad = torch.matmul(full_rank_grad,self.ortho_matrix.t())
        elif self.proj_type == 'right':
            if self.ortho_matrix is None or iter % self.update_proj_gap == 0:
                self.ortho_matrix = self.get_orthogonal_matrix(full_rank_grad, self.rank, type='right')
            low_rank_grad = torch.matmul(full_rank_grad, self.ortho_matrix.t())
        elif self.proj_type == 'left':
            if self.ortho_matrix is None or iter % self.update_proj_gap == 0:
                self.ortho_matrix = self.get_orthogonal_matrix(full_rank_grad, self.rank, type='left')
            low_rank_grad = torch.matmul(self.ortho_matrix.t(), full_rank_grad)
        elif self.proj_type == 'full':
            if self.ortho_matrix is None or iter % self.update_proj_gap == 0:
                self.ortho_matrix = self.get_orthogonal_matrix(full_rank_grad, self.rank, type='full')
            low_rank_grad = torch.matmul(self.ortho_matrix[0].t(), full_rank_grad) @ self.ortho_matrix[1].t()
                
        return low_rank_grad

    def project_back(self, low_rank_grad):

        if self.proj_type == 'std':
            if low_rank_grad.shape[0] >= low_rank_grad.shape[1]:
                full_rank_grad = torch.matmul(low_rank_grad, self.ortho_matrix)
            else:
                full_rank_grad = torch.matmul(self.ortho_matrix, low_rank_grad)
        elif self.proj_type == 'reverse_std':
            if low_rank_grad.shape[0] <= low_rank_grad.shape[1]: # note this is different from std
                full_rank_grad = torch.matmul(self.ortho_matrix, low_rank_grad)
            else:
                full_rank_grad = torch.matmul(low_rank_grad, self.ortho_matrix)
        elif self.proj_type == 'right':
            full_rank_grad = torch.matmul(low_rank_grad, self.ortho_matrix)
        elif self.proj_type == 'left':
            full_rank_grad = torch.matmul(self.ortho_matrix, low_rank_grad)
        elif self.proj_type == 'full':
            full_rank_grad = torch.matmul(self.ortho_matrix[0], low_rank_grad) @ self.ortho_matrix[1]
        
        
        return full_rank_grad * self.scale

    # svd decomposition
    def get_orthogonal_matrix(self, weights, rank, type):
        module_params = weights
        ret_val = None
        original_device, original_type = None, None

        if module_params.data.dtype != torch.float:
            float_data = False
            original_type = module_params.data.dtype
            original_device = module_params.data.device
            matrix = module_params.data.float()
        else:
            float_data = True
            matrix = module_params.data
        
        #print(f"Doing SVD decomposition. Matrix shape: {matrix.shape}")
        #start = time.time()
        global thread_pool
        if self.ortho_matrix is None:
            # first time, GPU SVD
            self.last_result = self.get_orthogonal_matrix_cpu(matrix, rank, type, float_data, original_device, original_type)
            ret_val = self.last_result
        else:
            if isinstance(self.last_result, Future):
                self.last_result = self.last_result.result()
                print("Fetch last result")
            if type == 'full':
                ret_val = [m.cuda() for m in self.last_result]
            else:
                ret_val = self.last_result.cuda()
            self.last_result = thread_pool.submit(self.get_orthogonal_matrix_cpu, matrix.cpu(), rank, type, float_data, original_device, original_type)
        #print(f"SVD decomposition done. Time taken: {time.time() - start:.2f} s")
        return ret_val
    
    # svd decomposition
    def get_orthogonal_matrix_cpu(self, matrix, rank, type, float_data, original_device, original_type):
        #print(f"Doing SVD decomposition CPU. Matrix shape: {matrix.shape}")
        #start = time.time()
        # CPU SVD
        U, s, Vh = torch.linalg.svd(matrix, full_matrices = False)
        #U, s, Vh = fbpca.pca(matrix.numpy(force=True).astype(np.float32), k=rank, n_iter=2)
        #U, s, Vh = torch.from_numpy(U), torch.from_numpy(s), torch.from_numpy(Vh)
        #print(f"SVD decomposition CPU done. Time taken: {time.time() - start:.2f} s")
        #make the smaller matrix always to be orthogonal matrix
        if type=='right':
            A = U[:, :rank] @ torch.diag(s[:rank])
            B = Vh[:rank, :]
            
            if not float_data:
                B = B.to(original_device).type(original_type)
            return B
        elif type=='left':
            A = U[:, :rank]
            B = torch.diag(s[:rank]) @ Vh[:rank, :]
            if not float_data:
                A = A.to(original_device).type(original_type)
            return A
        elif type=='full':
            A = U[:, :rank]
            B = Vh[:rank, :]
            if not float_data:
                A = A.to(original_device).type(original_type)
                B = B.to(original_device).type(original_type)
            return [A, B]
        else:
            raise ValueError('type should be left, right or full')

