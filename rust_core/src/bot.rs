use crate::grpc_client::swarm::{Action, Entity, Observation};
use azalea::entity::{LookDirection, Position};
use azalea::local_player::Hunger;
use azalea::prelude::*;
use azalea::{BlockPos, Client, Event};
use azalea_entity::metadata::Health;
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::{Mutex, mpsc};

use once_cell::sync::Lazy;

pub static PENDING_STATES: Lazy<Mutex<HashMap<String, SwarmState>>> =
    Lazy::new(|| Mutex::new(HashMap::new()));

#[derive(Clone, Component)]
pub struct SwarmState {
    pub agent_id: String,
    pub obs_tx: mpsc::Sender<Observation>,
    pub current_action: Arc<Mutex<Option<Action>>>,
    pub tick_counter: Arc<Mutex<u64>>, // YENİ: Cooldown sayacı
}

impl Default for SwarmState {
    fn default() -> Self {
        let (tx, _) = mpsc::channel(1);
        Self {
            agent_id: "DefaultBot".to_string(),
            obs_tx: tx,
            current_action: Arc::new(Mutex::new(None)),
            tick_counter: Arc::new(Mutex::new(0)),
        }
    }
}

pub async fn handle(bot: Client, event: Event, state: SwarmState) -> anyhow::Result<()> {
    match event {
        Event::Login => {
            println!(
                "Bot {} logged in! Awaiting ECS state injection...",
                bot.profile().name
            );
            let mut pending = PENDING_STATES.lock().await;
            let bot_name = bot.profile().name.clone();

            if let Some(real_state) = pending.remove(&bot_name) {
                bot.ecs
                    .write()
                    .entity_mut(bot.entity)
                    .insert(real_state.clone());
                println!("{} real State injected successfully!", real_state.agent_id);
            }
        }

        Event::Tick => {
            // ÖNEMLİ DÜZELTME 1: Bot oyundan düştüğünde PANIC atmasını engelle!
            let health_opt = bot.ecs.read().get::<Health>(bot.entity).map(|h| h.0);
            if health_opt.is_none() {
                return Ok(()); // Eğer can değeri yoksa (bot oyunda değilse) pas geç, çökme!
            }
            let health_val = health_opt.unwrap();

            // Eğer bot yaşıyorsa Tick sayacını (zamanı) ilerlet
            let current_tick = {
                let mut tc = state.tick_counter.lock().await;
                *tc += 1;
                *tc
            };

            let hunger_val = bot
                .ecs
                .read()
                .get::<Hunger>(bot.entity)
                .map(|h| h.food as f32)
                .unwrap_or(20.0);

            // 5×5×5 Voxel Radar
            let mut block_grid = Vec::with_capacity(125);
            {
                let bot_pos = BlockPos::from(bot.position());
                let world_lock = bot.world();
                let world = world_lock.read();

                for y in -2..=2 {
                    for z in -2..=2 {
                        for x in -2..=2 {
                            let check_pos = bot_pos + BlockPos::new(x, y, z);
                            block_grid.push(
                                world
                                    .get_block_state(check_pos)
                                    .map(|s| s.id() as i32)
                                    .unwrap_or(0),
                            );
                        }
                    }
                }
            }

            let bot_yaw = bot
                .ecs
                .read()
                .get::<LookDirection>(bot.entity)
                .map(|l| l.y_rot())
                .unwrap_or(0.0);
            let bot_pitch = bot
                .ecs
                .read()
                .get::<LookDirection>(bot.entity)
                .map(|l| l.x_rot())
                .unwrap_or(0.0);
            let bot_pos = bot.position();

            // En yakın 5 varlık
            let mut detected_entities = Vec::new();
            {
                let mut ecs = bot.ecs.write();
                let mut query = ecs.query::<(bevy_ecs::entity::Entity, &Position)>();

                for (entity, pos) in query.iter(&ecs) {
                    if entity == bot.entity {
                        continue;
                    }

                    let dx = pos.x - bot_pos.x;
                    let dy = pos.y - bot_pos.y;
                    let dz = pos.z - bot_pos.z;

                    let distance = (dx * dx + dy * dy + dz * dz).sqrt();
                    let target_yaw = (-dx).atan2(dz).to_degrees();
                    let mut rel_yaw = (target_yaw as f32) - bot_yaw;
                    while rel_yaw > 180.0 {
                        rel_yaw -= 360.0;
                    }
                    while rel_yaw < -180.0 {
                        rel_yaw += 360.0;
                    }

                    let horiz_dist = (dx * dx + dz * dz).sqrt();
                    let target_pitch = (-dy).atan2(horiz_dist).to_degrees();
                    let mut rel_pitch = (target_pitch as f32) - bot_pitch;
                    while rel_pitch > 180.0 {
                        rel_pitch -= 360.0;
                    }
                    while rel_pitch < -180.0 {
                        rel_pitch += 360.0;
                    }

                    let mut type_val = 0;
                    if ecs.get::<azalea_entity::metadata::Zombie>(entity).is_some()
                        || ecs
                            .get::<azalea_entity::metadata::Skeleton>(entity)
                            .is_some()
                        || ecs
                            .get::<azalea_entity::metadata::Creeper>(entity)
                            .is_some()
                        || ecs.get::<azalea_entity::metadata::Spider>(entity).is_some()
                    {
                        type_val = 1; // Düşman
                    } else if ecs.get::<azalea_entity::metadata::Player>(entity).is_some() {
                        type_val = 2; // Dost
                    }

                    detected_entities.push((
                        entity,
                        Entity {
                            entity_id: 0,
                            entity_type: type_val,
                            distance: distance as f32,
                            relative_yaw: rel_yaw,
                            relative_pitch: rel_pitch,
                        },
                    ));
                }
            }

            detected_entities.sort_by(|a, b| {
                a.1.distance
                    .partial_cmp(&b.1.distance)
                    .unwrap_or(std::cmp::Ordering::Equal)
            });

            let nearest_local: Vec<_> = detected_entities.into_iter().take(5).collect();
            let mut nearest_entities = Vec::new();
            for (_, proto_ent) in &nearest_local {
                nearest_entities.push(proto_ent.clone());
            }

            while nearest_entities.len() < 5 {
                nearest_entities.push(Entity {
                    entity_id: 0,
                    entity_type: 0,
                    distance: 999.0,
                    relative_yaw: 0.0,
                    relative_pitch: 0.0,
                });
            }

            let obs = Observation {
                agent_id: state.agent_id.clone(),
                health: health_val,
                hunger: hunger_val,
                position_x: bot_pos.x,
                position_y: bot_pos.y,
                position_z: bot_pos.z,
                dost_bana_vurdu: false,
                vuran_dost_yaw: 0.0,
                vuran_dost_pitch: 0.0,
                entities: nearest_entities,
                block_grid,
            };

            let _ = state.obs_tx.try_send(obs);

            // ── 2. Python'dan Gelen Action'ı Uygula ─────────────────────────
            let action_opt = state.current_action.lock().await.clone();
            if let Some(action) = action_opt {
                let forward = (action.key_bitmask & (1 << 0)) != 0;
                let backward = (action.key_bitmask & (1 << 1)) != 0;
                let left = (action.key_bitmask & (1 << 2)) != 0;
                let right = (action.key_bitmask & (1 << 3)) != 0;
                let jump = (action.key_bitmask & (1 << 4)) != 0;
                let attack = (action.key_bitmask & (1 << 6)) != 0;
                let interact = (action.key_bitmask & (1 << 7)) != 0;

                let direction = if forward && left {
                    azalea::WalkDirection::ForwardLeft
                } else if forward && right {
                    azalea::WalkDirection::ForwardRight
                } else if backward && left {
                    azalea::WalkDirection::BackwardLeft
                } else if backward && right {
                    azalea::WalkDirection::BackwardRight
                } else if forward {
                    azalea::WalkDirection::Forward
                } else if backward {
                    azalea::WalkDirection::Backward
                } else if left {
                    azalea::WalkDirection::Left
                } else if right {
                    azalea::WalkDirection::Right
                } else {
                    azalea::WalkDirection::None
                };

                bot.walk(direction);
                bot.set_jumping(jump);

                // — Attack: Gerçek Hedef Vurma —
                if attack {
                    if let Some((real_target, proto_info)) = nearest_local.first() {
                        if proto_info.distance < 4.0 && proto_info.entity_type == 1 {
                            // ÖNEMLİ DÜZELTME 2: COOLDOWN EKLENDİ!
                            // Bot saniyede 20 kere vurup sunucuyu çökertmesin diye
                            // Sadece her 10 tick'te (0.5 saniyede) bir vuracak.
                            if current_tick % 10 == 0 {
                                bot.attack(*real_target);
                            }
                        }
                    }
                }

                if interact {
                    bot.start_use_item();
                }

                let slot = (action.select_slot as u8).min(8);
                bot.set_selected_hotbar_slot(slot);

                let current_yaw = bot
                    .ecs
                    .read()
                    .get::<LookDirection>(bot.entity)
                    .map(|l| l.y_rot())
                    .unwrap_or(0.0);
                let current_pitch = bot
                    .ecs
                    .read()
                    .get::<LookDirection>(bot.entity)
                    .map(|l| l.x_rot())
                    .unwrap_or(0.0);

                let new_yaw = current_yaw + (action.delta_yaw * 10.0);
                let new_pitch = (current_pitch * 0.8 + action.delta_pitch * 5.0).clamp(-90.0, 90.0);

                bot.set_direction(new_yaw, new_pitch);
            }
        }

        Event::Chat(m) => {
            println!("[{}] Chat: {}", state.agent_id, m.message());
        }

        Event::Disconnect(reason) => {
            println!(
                "⚠️ [{}] Sunucudan koptu veya atıldı! Sebep: {:?}",
                state.agent_id, reason
            );
        }

        _ => {}
    }
    Ok(())
}
