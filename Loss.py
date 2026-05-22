def compute_distillation_loss(student_dist, teacher_dist, student_h, teacher_h):
    # 1. Action KL Divergence Loss
    # Công thức KL cho 2 phân phối chuẩn: KL(N_student || N_teacher)
    kl_div = torch.distributions.kl.kl_divergence(student_dist, teacher_dist).mean()
    
    # 2. Hidden State Alignment (MSE Loss)
    # Cần một projection layer nếu hidden_size của Teacher và Student khác nhau
    # Ở đây giả định dùng cùng hidden_size để triệt tiêu ma trận W
    mse_loss = nn.MSELoss()(student_h, teacher_h.detach())
    
    # Tổng hợp Loss
    beta = 0.5 # Hệ số cân bằng
    total_loss = kl_div + beta * mse_loss
    return total_loss

# --- Giả lập một bước huấn luyện (Training Step) ---
# Giả sử ta lấy một batch quỹ đạo (trajectories) độ dài T=32 từ buffer
batch_size, seq_len = 16, 32

# Dữ liệu giả lập
obs_seq = torch.randn(batch_size, seq_len, 8)       # Student chỉ thấy qpos
state_seq = torch.randn(batch_size, seq_len, 17)     # Teacher thấy cả qpos + qvel
llm_z_seq = torch.randn(batch_size, seq_len, 256)    # Vector chiến lược từ LLM

student = StudentInferenceActor(obs_dim=8, act_dim=6, hidden_size=128)
teacher = TeacherLLMActor(full_state_dim=17, llm_latent_dim=256, act_dim=6, hidden_size=128)

# Forward pass
stud_dist, stud_h, _ = student(obs_seq)
with torch.no_grad(): # Không cập nhật gradient cho Teacher trong bước này
    teach_dist, teach_h, _ = teacher(state_seq, llm_z_seq)

# Tính loss và backward
loss = compute_distillation_loss(stud_dist, teach_dist, stud_h, teach_h)
# optimizer.zero_grad()
# loss.backward()
# optimizer.step()