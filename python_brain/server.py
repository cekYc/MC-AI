import grpc
from concurrent import futures
import threading
import copy
import math
import os
import sys
import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Bernoulli, Normal
from torch.utils.tensorboard import SummaryWriter

try:
    import swarm_pb2
    import swarm_pb2_grpc
except ImportError:
    print("HATA: swarm_pb2 veya swarm_pb2_grpc bulunamadı.")
    sys.exit(1)

# ---------------------------------------------------------------------------
# SABITLER
# ---------------------------------------------------------------------------
OBS_DIM        = 465     # Son 3 adımın hafızası (155 * 3)
GAMMA          = 0.99
EPS_CLIP       = 0.2
ENTROPY_COEF   = 0.001   # Rastgeleliği azalttık, artık ne öğrendiyse uygulayacak!
UPDATE_TIMESTEP = 1000
K_EPOCHS       = 4
LR             = 3e-4

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------------------------------------------------------------------------
# YAPAY ZEKA MODELİ
# ---------------------------------------------------------------------------
class SwarmBrain(nn.Module):
    def __init__(self, obs_dim: int = OBS_DIM):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(obs_dim, 256), nn.ReLU(),
            nn.Linear(256, 128),    nn.ReLU(),
        )
        # 7 Tuş (İleri, Geri, Sol, Sağ, Zıpla, Eğil, VUR)
        self.actor_buttons      = nn.Linear(128, 7) 
        self.actor_mouse_mean   = nn.Linear(128, 2)
        self.actor_mouse_logstd = nn.Parameter(torch.zeros(1, 2))
        self.critic             = nn.Linear(128, 1)

    def forward(self, obs: torch.Tensor):
        x           = self.shared(obs)
        button_logits = self.actor_buttons(x)
        mouse_mean  = self.actor_mouse_mean(x)
        mouse_std   = self.actor_mouse_logstd.exp().expand_as(mouse_mean)
        value       = self.critic(x)
        return button_logits, mouse_mean, mouse_std, value

global_brain = SwarmBrain(obs_dim=OBS_DIM).to(device)

if os.path.exists("brain.pth"):
    try:
        raw = torch.load("brain.pth", weights_only=True)
        global_brain.load_state_dict(raw)
        print("💾 Kayıtlı beyin (brain.pth) yüklendi!")
    except Exception as e:
        print("⚠️ Eski beyin boyutu uyumsuz! Lütfen 'brain.pth' dosyasını silin.")
        sys.exit(1)
else:
    print("🌱 Yeni bir beyin oluşturuldu. Sıfırdan öğrenmeye başlıyor...")

optimizer = torch.optim.Adam(global_brain.parameters(), lr=LR)
writer = SummaryWriter("runs/swarm_deney_avci")

training_lock = threading.Lock()
is_training   = False
global_step   = 0

DANGER_BLOCKS  = {10, 11, 51, 81}
VALUABLE_BLOCKS = {17, 162, 56, 57}

def normalize_block(block_id: int) -> float:
    if block_id == 0:               return  0.0
    if block_id in DANGER_BLOCKS:   return -1.0
    if block_id in VALUABLE_BLOCKS: return  1.0
    return 0.5

# ---------------------------------------------------------------------------
# HAFIZA VE ÖDÜL SİSTEMİ
# ---------------------------------------------------------------------------
class AgentMemory:
    def __init__(self):
        self.states = []
        self.actions_buttons = []
        self.actions_mouse = []
        self.logprobs_buttons = []
        self.logprobs_mouse = []
        self.rewards = []
        self.values = []
        self.dones = []
        
        self.last_health = 20.0
        self.last_hunger = 20.0
        self.last_pos = None
        self.spawn_pos = None
        self.obs_history = [] # 3 Saniyelik Hafıza

    def clear(self):
        for attr in ("states", "actions_buttons", "actions_mouse",
                     "logprobs_buttons", "logprobs_mouse",
                     "rewards", "values", "dones"):
            getattr(self, attr).clear()

    def calculate_reward(self, health: float, hunger: float, pos: tuple, btn_actions=None, entities=None) -> tuple[float, bool]:
        reward  = -0.01
        is_done = False

        if self.last_health > 0.0 and health <= 0.0:
            reward  -= 5.0
            is_done  = True
            self.last_health = health
            self.last_hunger = hunger
            return reward, is_done

        # Göç Ödülü
        if self.spawn_pos is None:
            self.spawn_pos = pos
        else:
            dist_from_spawn = math.sqrt((pos[0] - self.spawn_pos[0])**2 + (pos[2] - self.spawn_pos[2])**2)
            reward += (dist_from_spawn * 0.001)

        # Hareket Ödülü
        if self.last_pos is not None:
            dx = pos[0] - self.last_pos[0]
            dy = pos[1] - self.last_pos[1]
            dz = pos[2] - self.last_pos[2]
            dist = math.sqrt(dx*dx + dy*dy + dz*dz)
            if dist > 0.5:
                reward += 0.5
                self.last_pos = pos
            elif dist < 0.05:
                reward -= 0.05
        else:
            self.last_pos = pos

        # SALDIRI ÖDÜLÜ
        if btn_actions is not None and entities is not None:
            if btn_actions[6] > 0.5: # 7. tuş (Saldırı)
                for ent in entities:
                    if ent.entity_type == 1 and ent.distance < 4.0: # Düşmana salladıysa
                        reward += 0.5 
                        break

        delta_health = health - self.last_health
        if delta_health < 0:
            reward += delta_health * 5.0
        elif delta_health > 0:
            if self.last_health > 0.0 and delta_health <= 15.0:
                reward += delta_health * 1.0

        if hunger < 6.0:
            reward -= 0.1

        self.last_health = health
        self.last_hunger = hunger
        return reward, is_done

# ---------------------------------------------------------------------------
# EĞİTİM DÖNGÜSÜ
# ---------------------------------------------------------------------------
def train_ppo(memory: AgentMemory, agent_id: str):
    global is_training, global_step
    with training_lock:
        is_training = True
        try:
            old_states   = torch.tensor(np.array(memory.states), dtype=torch.float32).to(device)
            old_btn      = torch.tensor(np.array(memory.actions_buttons), dtype=torch.float32).to(device)
            old_mouse    = torch.tensor(np.array(memory.actions_mouse), dtype=torch.float32).to(device)
            old_lp_btn   = torch.tensor(np.array(memory.logprobs_buttons), dtype=torch.float32).to(device)
            old_lp_mouse = torch.tensor(np.array(memory.logprobs_mouse), dtype=torch.float32).to(device)

            rewards_list = []
            discounted   = 0.0
            for reward, done in zip(reversed(memory.rewards), reversed(memory.dones)):
                if done: discounted = 0.0
                discounted = reward + GAMMA * discounted
                rewards_list.insert(0, discounted)

            rewards_t = torch.tensor(rewards_list, dtype=torch.float32).to(device)
            rewards_t = (rewards_t - rewards_t.mean()) / (rewards_t.std() + 1e-7)

            for _ in range(K_EPOCHS):
                btn_logits, m_mean, m_std, state_values = global_brain(old_states)

                btn_probs    = torch.sigmoid(btn_logits)
                dist_btn     = Bernoulli(btn_probs)
                lp_btn_now   = dist_btn.log_prob(old_btn).sum(dim=1)
                entropy_btn  = dist_btn.entropy().sum(dim=1).mean()

                dist_mouse   = Normal(m_mean, m_std)
                lp_mouse_now = dist_mouse.log_prob(old_mouse).sum(dim=1)
                entropy_mouse = dist_mouse.entropy().sum(dim=1).mean()

                advantages   = rewards_t - state_values.squeeze(-1).detach()
                ratios = (torch.exp(lp_btn_now - old_lp_btn) + torch.exp(lp_mouse_now - old_lp_mouse)) / 2.0
                surr1 = ratios * advantages
                surr2 = torch.clamp(ratios, 1 - EPS_CLIP, 1 + EPS_CLIP) * advantages

                actor_loss  = -torch.min(surr1, surr2).mean()
                critic_loss = 0.5 * nn.MSELoss()(state_values.squeeze(-1), rewards_t)
                entropy_bonus = ENTROPY_COEF * (entropy_btn + entropy_mouse)

                loss = actor_loss + critic_loss - entropy_bonus

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(global_brain.parameters(), 0.5)
                optimizer.step()

            global_step += 1
            avg_reward   = sum(memory.rewards) / max(len(memory.rewards), 1)

            writer.add_scalar("1_Performans/Ortalama_Odul", avg_reward, global_step)
            writer.add_scalar("2_Beyin/Kayip_Loss", loss.item(), global_step)
            writer.add_scalar("2_Beyin/Deger_Tahmini_Critic", state_values.mean().item(), global_step)
            
            torch.save(global_brain.state_dict(), "brain.pth")
            print(f"✅ [{agent_id}] Eğitim tamamlandı. Kayıp: {loss.item():.4f}")
        finally:
            is_training = False
    memory.clear()

# ---------------------------------------------------------------------------
# GRPC BAĞLANTISI
# ---------------------------------------------------------------------------
class BrainServiceServicer(swarm_pb2_grpc.BrainServiceServicer):
    def StreamActions(self, request_iterator, context):
        agent_id   = "Bilinmeyen"
        memory     = AgentMemory()
        step_count = 0
        try:
            for obs in request_iterator:
                agent_id = obs.agent_id
                pos = (obs.position_x, obs.position_y, obs.position_z)

                block_grid = [normalize_block(b) for b in obs.block_grid]
                if len(block_grid) < 125: block_grid += [0.0] * (125 - len(block_grid))

                entities_flat = []
                for ent in obs.entities:
                    entities_flat.extend([
                        0.0,
                        float(ent.entity_type),
                        min(ent.distance / 30.0, 1.0),
                        ent.relative_yaw / 180.0,
                        ent.relative_pitch / 180.0,
                    ])

                current_obs = [obs.health / 20.0, obs.hunger / 20.0] + block_grid + entities_flat + [
                    1.0 if obs.dost_bana_vurdu else 0.0, 
                    obs.vuran_dost_yaw / 180.0, 
                    obs.vuran_dost_pitch / 180.0
                ]
                
                # Hafıza Yığınağı (Frame Stacking)
                memory.obs_history.append(current_obs)
                if len(memory.obs_history) > 3:
                    memory.obs_history.pop(0)

                stacked_obs = []
                for idx in range(3):
                    if idx < len(memory.obs_history):
                        stacked_obs.extend(memory.obs_history[idx])
                    else:
                        stacked_obs.extend(current_obs)

                obs_tensor = torch.tensor(stacked_obs, dtype=torch.float32).unsqueeze(0).to(device)

                with torch.no_grad():
                    btn_logits, m_mean, m_std, value = global_brain(obs_tensor)
                    btn_probs   = torch.sigmoid(btn_logits)
                    dist_btn    = Bernoulli(btn_probs)
                    btn_sample  = dist_btn.sample()
                    btn_actions = btn_sample.squeeze(0).cpu().numpy()
                    lp_btn      = dist_btn.log_prob(btn_sample).sum(dim=1).squeeze(0).cpu().numpy()

                    dist_mouse   = Normal(m_mean, m_std)
                    mouse_sample = dist_mouse.sample()
                    mouse_actions = mouse_sample.squeeze(0).cpu().numpy()
                    lp_mouse     = dist_mouse.log_prob(mouse_sample).sum(dim=1).squeeze(0).cpu().numpy()

                step_reward, is_done = memory.calculate_reward(obs.health, obs.hunger, pos, btn_actions, obs.entities)

                memory.states.append(stacked_obs)
                memory.actions_buttons.append(btn_actions)
                memory.actions_mouse.append(mouse_actions)
                memory.logprobs_buttons.append(lp_btn)
                memory.logprobs_mouse.append(lp_mouse)
                memory.rewards.append(step_reward)
                memory.values.append(value.item())
                memory.dones.append(is_done)

                step_count += 1

                if len(memory.states) >= UPDATE_TIMESTEP and not is_training:
                    memory_copy = copy.deepcopy(memory)
                    memory.clear()
                    step_count = 0
                    threading.Thread(target=train_ppo, args=(memory_copy, agent_id), daemon=True).start()

                # 7 TUŞ OKUMA DÖNGÜSÜ
                key_mask = 0
                for i in range(7):
                    if btn_actions[i] > 0.5:
                        key_mask |= (1 << i)

                action = swarm_pb2.Action(
                    key_bitmask=int(key_mask),
                    select_slot=0,
                    delta_yaw=float(mouse_actions[0]),
                    delta_pitch=float(mouse_actions[1]),
                )
                yield action
        except Exception as exc:
            print(f"⚠️ [{agent_id}] Koptu: {exc}")

def serve():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=20))
    swarm_pb2_grpc.add_BrainServiceServicer_to_server(BrainServiceServicer(), server)
    server.add_insecure_port("[::]:50051")
    server.start()
    print("📡 Beyin aktif. Port 50051'de bağlantı bekleniyor...")
    try:
        server.wait_for_termination()
    except KeyboardInterrupt:
        server.stop(0)

if __name__ == "__main__":
    serve()