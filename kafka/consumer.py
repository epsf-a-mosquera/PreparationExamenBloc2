from kafka import KafkaConsumer


kafka_consumer = KafkaConsumer(
    "test",
    bootstrap_servers="localhost:9092",
    auto_offset_reset="earliest"
)


for message in kafka_consumer:
    print(message.value.decode("utf-8"))

