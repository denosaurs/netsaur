use netsaur::{
    array, Activation, BackendType, Cost, Dataset, Dense, Layer, Network, NetworkConfig,
};

fn main() {
    let mut network = Network::new(NetworkConfig {
        layers: vec![
            Layer::Dense(Dense {
                size: 3,
                activation: Some(Activation::Sigmoid),
            }),
            Layer::Dense(Dense {
                size: 1,
                activation: Some(Activation::Sigmoid),
            }),
        ],
        size: &[4, 2],
        backend: BackendType::CPU,
        cost: Cost::MSE,
    });

    network.train(
        vec![Dataset {
            inputs: array![[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]].into_dyn(),
            outputs: array![[0.0], [1.0], [1.0], [0.0]].into_dyn(),
        }],
        10000,
        0.1,
    );

    println!("{:#?}", network.predict(array![[0.0, 0.0]].into_dyn()));
    println!("{:#?}", network.predict(array![[1.0, 0.0]].into_dyn()));
    println!("{:#?}", network.predict(array![[0.0, 1.0]].into_dyn()));
    println!("{:#?}", network.predict(array![[1.0, 1.0]].into_dyn()));
}
