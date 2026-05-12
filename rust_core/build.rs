fn main() -> Result<(), Box<dyn std::error::Error>> {
    unsafe {
        std::env::set_var("PROTOC", protoc_bin_vendored::protoc_bin_path().unwrap());
    }
    tonic_prost_build::configure()
        .build_server(false)
        .build_client(true)
        .compile_protos(&["../shared/swarm.proto"], &["../shared"])?;
    Ok(())
}
