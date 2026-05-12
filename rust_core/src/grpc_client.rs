pub mod swarm {
    tonic::include_proto!("swarm");
}

use std::sync::Arc;
use swarm::brain_service_client::BrainServiceClient;
use swarm::{Action, Observation};
use tokio::sync::Mutex;
use tokio::sync::mpsc;
use tokio_stream::wrappers::ReceiverStream;

/// Her bot için bağımsız bir gRPC link tutar.
/// DÜZELTME #1: main.rs'te artık her bot için ayrı bir GrpcLink oluşturuluyor;
/// bu yapı zaten tek-bot tasarımına uygundu, değiştirilmedi.
pub struct GrpcLink {
    pub obs_tx: mpsc::Sender<Observation>,
    pub current_action: Arc<Mutex<Option<Action>>>,
}

pub async fn start_grpc_client(server_addr: String) -> anyhow::Result<GrpcLink> {
    let mut client = BrainServiceClient::connect(server_addr).await?;

    let (obs_tx, obs_rx) = mpsc::channel(100);
    let current_action = Arc::new(Mutex::new(None));
    let action_writer = current_action.clone();

    tokio::spawn(async move {
        let request = tonic::Request::new(ReceiverStream::new(obs_rx));
        match client.stream_actions(request).await {
            Ok(response) => {
                let mut stream = response.into_inner();
                while let Ok(Some(action)) = stream.message().await {
                    *action_writer.lock().await = Some(action);
                }
            }
            Err(e) => {
                eprintln!("gRPC Stream Error: {:?}", e);
            }
        }
    });

    Ok(GrpcLink {
        obs_tx,
        current_action,
    })
}
