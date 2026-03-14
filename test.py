from qiskit_ibm_runtime import QiskitRuntimeService

token = "CFgeReU6HmU7KikUDmrCgoihh73yWXihv7VeWkvCg3vo"   # regenerate first!
crn   = "crn:v1:bluemix:public:quantum-computing:us-east:a/3c536508f7ef4a99bdec8920107e9e7a:d8e72eda-30e2-4a04-8d14-6f85591eff36::"

service = QiskitRuntimeService(
    channel="ibm_cloud",
    token=token,
    instance=crn
)

backends = service.backends()
print(f"✅ Connected! {len(backends)} backends available:")
for b in backends:
    print(f"  {b.name} | {b.num_qubits} qubits")