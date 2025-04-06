import torch
import torch.func as func


class MesonExchangePotential(torch.nn.Module):
    def __init__(self, g_att, g_rep, m_pi, m_rho, r_cutoff, r_core, device):
        super().__init__()
        self.g_att = g_att
        self.g_rep = g_rep
        self.m_pi = m_pi
        self.m_rho = m_rho

        self.r_cutoff = r_cutoff
        self.r_core = r_core

        self.device = device

        self.g_att_sq = self.g_att ** 2
        self.g_rep_sq = self.g_rep ** 2

        self._compute_compiled = torch.compile(self.compute)
        self._compute_derivatives_compiled = torch.compile(self._compute_derivatives_impl)
        self._compute_force_only_compiled = torch.compile(self._compute_force_only_impl)

    def compute(self, r):
        r_safe = torch.clamp(r, min=1e-10)

        yukawa = -self.g_att_sq * torch.exp(-self.m_pi * r_safe) / r_safe

        repulsion = self.g_rep_sq * torch.exp(-self.m_rho * r_safe) / r_safe

        return torch.where(
            r_safe < self.r_cutoff,
            yukawa + repulsion,
            torch.zeros_like(r)
        )

    def compute_force(self, r):
        r_tensor = r.detach().clone().requires_grad_(True)
        potential = self.compute(r_tensor)
        force = -torch.autograd.grad(potential.sum(), r_tensor, create_graph=True)[0]
        return force

    def compute_force_only(self, positions):
        """
        Вычисление только сил без производных
        
        Args:
            positions: Позиции частиц [N, 3]
            
        Returns:
            torch.Tensor: Силы, действующие на частицы [N, 3]
        """
        return self._compute_force_only_compiled(positions)

    def _compute_force_only_impl(self, positions):
        """
        Вычисление только сил без производных, используя автоматическое дифференцирование
        
        Args:
            positions: Позиции частиц [N, 3]
            
        Returns:
            torch.Tensor: Силы, действующие на частицы [N, 3]
        """
        # Определяем функцию потенциальной энергии системы
        def potential_energy(pos):
            r_ij = pos.unsqueeze(1) - pos.unsqueeze(0)
            distances = torch.norm(r_ij, dim=2)
            r_safe = torch.clamp(distances, min=1e-10)

            yukawa = -self.g_att_sq * torch.exp(-self.m_pi * r_safe) / r_safe
            repulsion = self.g_rep_sq * torch.exp(-self.m_rho * r_safe) / r_safe

            pair_potential = torch.where(
                (r_safe > self.r_core) & (r_safe < self.r_cutoff),
                yukawa + repulsion,
                torch.zeros_like(distances))

            return torch.sum(torch.triu(pair_potential, diagonal=1))
            
        # Вычисляем силы как отрицательный градиент потенциальной энергии
        forces = -func.grad(potential_energy)(positions)
        
        return forces

    def _compute_derivatives_impl(self, positions):
        N = positions.shape[0]
        r_ij = positions.unsqueeze(1) - positions.unsqueeze(0)
        distances = torch.norm(r_ij, dim=2)
        mask = (distances > self.r_core) & (distances < self.r_cutoff)

        if not torch.any(mask):
            return (
                torch.zeros_like(positions),
                torch.zeros((N, N, 3, 3), device=self.device),
                torch.zeros((N, 3, 3, 3), device=self.device)
            )

        def potential_energy(pos):
            r_ij = pos.unsqueeze(1) - pos.unsqueeze(0)
            distances = torch.norm(r_ij, dim=2)
            r_safe = torch.clamp(distances, min=1e-10)

            yukawa = -self.g_att_sq * torch.exp(-self.m_pi * r_safe) / r_safe
            repulsion = self.g_rep_sq * torch.exp(-self.m_rho * r_safe) / r_safe

            pair_potential = torch.where(
                (r_safe > self.r_core) & (r_safe < self.r_cutoff),
                yukawa + repulsion,
                torch.zeros_like(distances))

            return torch.sum(torch.triu(pair_potential, diagonal=1))

        forces = -func.grad(potential_energy)(positions)
        hessian = func.hessian(potential_energy)(positions)
        f_prime = hessian.reshape(N, 3, N, 3).permute(0, 2, 1, 3)
        f_double_prime = torch.zeros((N, 3, 3, 3), device=self.device)

        for i in range(N):
            def force_i(pos):
                all_pos = positions.clone()
                all_pos[i] = pos
                return -func.grad(potential_energy)(all_pos)[i]

            f_double_prime[i] = func.jacrev(func.jacrev(force_i))(positions[i])

        return forces, f_prime, f_double_prime

    def compute_derivatives(self, positions):
        return self._compute_derivatives_compiled(positions)

    def compute_energy_per_particle(self, positions):
        distances = torch.cdist(positions, positions)
        potential_matrix = self._compute_compiled(distances)
        mask = (distances > self.r_core) & (distances < self.r_cutoff)
        potential_matrix = potential_matrix * mask
        return torch.sum(potential_matrix, dim=1) / 2