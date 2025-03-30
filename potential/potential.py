import torch
import torch.func as func


class ModifiedYukawaPotential(torch.nn.Module):
    def __init__(self, V0, alpha, r1, V_rep, beta, r_core, device):
        """
        Модифицированный потенциал Юкавы с отталкивающей частью
        
        Args:
            V0 (float): Глубина притягивающей части потенциала
            alpha (float): Параметр затухания притягивающей части
            r1 (float): Радиус обрезания потенциала
            V_rep (float): Сила отталкивающей части потенциала
            beta (float): Параметр затухания отталкивающей части (больше alpha)
            r_core (float): Радиус кора (отталкивающей части)
            device (str): Устройство для вычислений ('cpu' или 'cuda')
        """
        super().__init__()
        self.V0 = V0
        self.alpha = alpha
        self.r1 = r1
        self.V_rep = V_rep
        self.beta = beta
        self.r_core = r_core
        self.device = device
        
        self._compute_compiled = torch.compile(self.compute)
        self._compute_derivatives_compiled = torch.compile(self._compute_derivatives_impl)

    def compute(self, r):
        """
        Вычисляет потенциал для заданного расстояния
        
        Args:
            r (torch.Tensor): Расстояние между частицами
            
        Returns:
            torch.Tensor: Значение потенциала
        """
        r_safe = torch.clamp(r, min=1e-10)
        
        attractive = self.V0 * torch.exp(-self.alpha * r_safe) / r_safe
        
        repulsive = self.V_rep * torch.exp(-self.beta * r_safe) / r_safe
        
        return torch.where(
            r_safe < self.r1,
            repulsive - attractive,  # Отталкивание - притяжение
            torch.zeros_like(r)
        )

    def compute_force(self, r):
        """
        Вычисляет силу для заданного расстояния, используя автоматическое дифференцирование
        
        Args:
            r (torch.Tensor): Расстояние между частицами
            
        Returns:
            torch.Tensor: Значение силы
        """
        r_tensor = r.detach().clone().requires_grad_(True)
        potential = self.compute(r_tensor)
        
        force = -torch.autograd.grad(
            potential.sum(), r_tensor, create_graph=True
        )[0]
        
        return force

    def _compute_derivatives_impl(self, positions):
        """
        Внутренняя реализация для вычисления производных потенциала
        """
        N = positions.shape[0]
        
        r_ij = positions.unsqueeze(1) - positions.unsqueeze(0)  # [N, N, 3]
        distances = torch.norm(r_ij, dim=2)  # [N, N]
        
        mask = (distances > 1e-10) & (distances < self.r1)
        
        if not torch.any(mask):
            return (
                torch.zeros_like(positions),
                torch.zeros((N, 3, 3), device=self.device),
                torch.zeros((N, 3, 3, 3), device=self.device)
            )
        
        def potential_energy(pos):
            r_ij = pos.unsqueeze(1) - pos.unsqueeze(0)  # [N, N, 3]
            distances = torch.norm(r_ij, dim=2)  # [N, N]
            mask = (distances > 1e-10) & (distances < self.r1)
            
            r_safe = torch.clamp(distances, min=1e-10)
            attractive = self.V0 * torch.exp(-self.alpha * r_safe) / r_safe
            repulsive = self.V_rep * torch.exp(-self.beta * r_safe) / r_safe
            pair_potential = torch.where(mask, repulsive - attractive, torch.zeros_like(distances))
            
            return torch.sum(torch.triu(pair_potential, diagonal=1))
        
        forces = -func.grad(potential_energy)(positions)
        
        hessian = func.hessian(potential_energy)(positions)
        f_prime = hessian.reshape(N, 3, N, 3).permute(0, 2, 1, 3)
        
        def compute_third_derivatives(pos):
            return func.jacfwd(func.jacrev(func.grad(potential_energy)))(pos)
        
        f_double_prime = compute_third_derivatives(positions)
        f_double_prime = f_double_prime.reshape(N, 3, N, 3, N, 3)
        f_double_prime_diag = torch.zeros((N, 3, 3, 3), device=self.device)
        for i in range(N):
            f_double_prime_diag[i] = f_double_prime[i, :, i, :, i, :]
        
        return forces, f_prime, f_double_prime_diag

    def compute_derivatives(self, positions):
        """
        Вычисляет производные потенциала для всех частиц
        
        Args:
            positions (torch.Tensor): Позиции частиц [N, 3]
            
        Returns:
            tuple: (силы, вторые производные, третьи производные)
        """
        return self._compute_derivatives_compiled(positions)

    def compute_energy_per_particle(self, positions):
        """
        Вычисляет потенциальную энергию для каждой частицы
        
        Args:
            positions (torch.Tensor): Позиции частиц [N, 3]
            
        Returns:
            torch.Tensor: Потенциальная энергия каждой частицы [N]
        """
        N = positions.shape[0]
        
        distances = torch.cdist(positions, positions)
        
        potential_matrix = self._compute_compiled(distances)
        
        mask = torch.ones_like(potential_matrix) - torch.eye(N, device=self.device)
        potential_matrix = potential_matrix * mask
        
        return torch.sum(potential_matrix, dim=1) / 2
