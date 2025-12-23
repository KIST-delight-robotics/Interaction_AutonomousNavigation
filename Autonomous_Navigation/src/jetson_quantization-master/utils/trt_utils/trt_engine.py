import tensorrt as trt
import pycuda.driver as cuda
import numpy as np

class TRTEngine:
    def __init__(self, engine_path):
        """TensorRT 엔진을 로드하고 초기화"""
        self.logger = trt.Logger(trt.Logger.WARNING)
        
        # 엔진 로드
        with open(engine_path, 'rb') as f:
            engine_data = f.read()
        
        runtime = trt.Runtime(self.logger)
        self.engine = runtime.deserialize_cuda_engine(engine_data)
        self.context = self.engine.create_execution_context()
        
        # 입력/출력 바인딩 설정
        self.inputs = []
        self.outputs = []
        self.bindings = []
        
        for i in range(self.engine.num_io_tensors):
            tensor_name = self.engine.get_tensor_name(i)
            dtype = trt.nptype(self.engine.get_tensor_dtype(tensor_name))
            shape = self.engine.get_tensor_shape(tensor_name)
            size = trt.volume(shape)
            
            # GPU 메모리 할당
            device_mem = cuda.mem_alloc(size * dtype().itemsize)
            self.bindings.append(int(device_mem))
            
            if self.engine.get_tensor_mode(tensor_name) == trt.TensorIOMode.INPUT:
                self.inputs.append({
                    'name': tensor_name,
                    'dtype': dtype,
                    'shape': shape,
                    'device_mem': device_mem
                })
            else:
                self.outputs.append({
                    'name': tensor_name,
                    'dtype': dtype,
                    'shape': shape,
                    'device_mem': device_mem
                })
        # print(f"\n\tINPUTS: {self.inputs}\n")
    
    def infer(self, input_data):
        """추론 실행"""
        # 중요: 입력 텐서 shape 설정
        self.context.set_input_shape(self.inputs[0]['name'], input_data.shape)

        # 입력 데이터를 GPU로 복사
        cuda.memcpy_htod(self.inputs[0]['device_mem'], input_data.astype(self.inputs[0]['dtype']))
        
        # 추론 실행
        self.context.execute_v2(bindings=self.bindings)
        # self.context.enqueueV3(bindings=self.bindings)
        
        # 출력 데이터를 CPU로 복사
        outputs = []
        for output in self.outputs:
            host_mem = np.empty(output['shape'], dtype=output['dtype'])
            cuda.memcpy_dtoh(host_mem, output['device_mem'])
            outputs.append(host_mem)
        
        return outputs
    
class TRTEngine_async:
    def __init__(self, engine_path):
        self.logger = trt.Logger(trt.Logger.WARNING)
        with open(engine_path, 'rb') as f:
            engine_data = f.read()
        runtime = trt.Runtime(self.logger)
        self.engine = runtime.deserialize_cuda_engine(engine_data)
        self.context = self.engine.create_execution_context()

        self.inputs = []
        self.outputs = []
        self.bindings = []

        for i in range(self.engine.num_io_tensors):
            tensor_name = self.engine.get_tensor_name(i)
            dtype = trt.nptype(self.engine.get_tensor_dtype(tensor_name))
            shape = self.engine.get_tensor_shape(tensor_name)
            size = trt.volume(shape)
            device_mem = cuda.mem_alloc(size * dtype().itemsize)
            self.bindings.append(int(device_mem))
            if self.engine.get_tensor_mode(tensor_name) == trt.TensorIOMode.INPUT:
                self.inputs.append({'name': tensor_name, 'dtype': dtype, 'shape': shape, 'device_mem': device_mem})
            else:
                self.outputs.append({'name': tensor_name, 'dtype': dtype, 'shape': shape, 'device_mem': device_mem})

        self.stream = cuda.Stream()  # 스트림 추가

    def infer(self, input_data):
        # 입력 shape 설정
        self.context.set_input_shape(self.inputs[0]['name'], input_data.shape)
        # 입력 데이터 GPU로 복사 (비동기)
        cuda.memcpy_htod_async(self.inputs[0]['device_mem'], input_data.astype(self.inputs[0]['dtype']), self.stream)
        # Tensor address 연결 (TRT 9.x 이상, enqueueV3 방식 필수)
        self.context.set_tensor_address(self.inputs[0]['name'], int(self.inputs[0]['device_mem']))
        for output in self.outputs:
            self.context.set_tensor_address(output['name'], int(output['device_mem']))
        # 추론 비동기 실행
        self.context.execute_async_v3(stream_handle=self.stream.handle)
        # 출력 데이터 CPU로 복사 (비동기)
        outputs = []
        for output in self.outputs:
            host_mem = np.empty(output['shape'], dtype=output['dtype'])
            cuda.memcpy_dtoh_async(host_mem, output['device_mem'], self.stream)
            outputs.append(host_mem)
        # 스트림 동기화 (필수)
        self.stream.synchronize()
        return outputs