mod bot;
pub mod grpc_client;

use azalea::prelude::*;
use bot::{PENDING_STATES, SwarmState, handle};
use grpc_client::start_grpc_client;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("Starting Project Swarm - Spinal Cord (Rust Client)");

    let bot_count = 1;
    println!("Connecting {} Swarm Bots to Minecraft Server...", bot_count);

    let mut swarm_builder = azalea::swarm::SwarmBuilder::new();

    for i in 1..=bot_count {
        let agent_id = format!("SwarmBot_{:02}", i);

        // DÜZELTME #1: Her bot kendi bağımsız gRPC stream'ini açıyor.
        // Böylece Python tarafındaki StreamActions, tek bota karşılık gelir
        // ve action routing karışıklığı tamamen ortadan kalkar.
        let grpc_link = match start_grpc_client("http://127.0.0.1:50051".to_string()).await {
            Ok(link) => {
                println!("[{}] gRPC bağlantısı kuruldu.", agent_id);
                link
            }
            Err(e) => {
                println!(
                    "[{}] gRPC bağlantısı başarısız: {}. Dummy mode.",
                    agent_id, e
                );
                let (tx, _rx) = tokio::sync::mpsc::channel(128);
                grpc_client::GrpcLink {
                    obs_tx: tx,
                    current_action: std::sync::Arc::new(tokio::sync::Mutex::new(None)),
                }
            }
        };

        let real_state = SwarmState {
            agent_id: agent_id.clone(),
            obs_tx: grpc_link.obs_tx,
            current_action: grpc_link.current_action,
            tick_counter: std::sync::Arc::new(tokio::sync::Mutex::new(0)),
        };

        {
            let mut pending = PENDING_STATES.lock().await;
            pending.insert(agent_id.clone(), real_state);
        }

        let account = Account::offline(&agent_id);
        swarm_builder = swarm_builder.add_account(account);
    }

    swarm_builder
        .set_handler(handle)
        .start("127.0.0.1:25565")
        .await;

    Ok(())
}
