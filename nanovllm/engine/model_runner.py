import pickle
import torch
import torch.distributed as dist
from multiprocessing.synchronize import Event
from multiprocessing.shared_memory import SharedMemory

from nanovllm.config import Config
from nanovllm.engine.sequence import Sequence
from nanovllm.utils.context import set_context, get_context, reset_context
from nanovllm.utils.memory import get_gpu_memory
from nanovllm.models.qwen3 import Qwen3ForCausalLM
from nanovllm.layers.sampler import Sampler
from nanovllm.utils.loader import load_model


class ModelRunner:

    def __init__(self, config: Config, rank: int, event: Event | list[Event]):
        self.config = config
        hf_config = config.hf_config
        self.block_size = config.kvcache_block_size
        self.enforce_eager = config.enforce_eager
        self.world_size = config.tensor_parallel_size
        self.rank = rank
        self.event = event
        """
        ✅ 参数说明：
        参数名	含义	示例
        backend	使用的通信后端（如 NCCL、Gloo、MPI 等）	"nccl"
        init_method	进程组如何建立连接的方法	"tcp://localhost:2333"
        world_size	总共参与训练/推理的进程数（即 GPU 数量）	4
        rank	当前进程的唯一编号（从 0 开始）	0, 1, 2, 3
        """
        dist.init_process_group("nccl", "tcp://localhost:2333", world_size=self.world_size, rank=rank)

        """
        分布式训练：在进行分布式训练时，不同节点上的进程需要访问不同的GPU资源。
        在这种情况下，可以根据进程的rank（通常是全局唯一的标识符，表示该进程在整个分布式设置中的角色）来为其分配相应的GPU。
        这样做可以避免多个进程竞争同一块GPU资源的情况，有助于更高效地利用硬件资源。
        """
        torch.cuda.set_device(rank)
        
        default_dtype = torch.get_default_dtype()
        torch.set_default_dtype(hf_config.torch_dtype)
        torch.set_default_device("cuda")
        self.model = Qwen3ForCausalLM(hf_config)
        load_model(self.model, config.model)
        self.sampler = Sampler()
        self.allocate_kv_cache(config.gpu_memory_utilization)
        if not self.enforce_eager:
            self.capture_cudagraph()
        torch.set_default_device("cpu")
        torch.set_default_dtype(default_dtype)

        """
        这段代码是一个典型的多进程分布式程序中的初始化逻辑，通常用于在多个进程中共享数据（如模型权重、缓存等）
        在多进程环境下，由 rank == 0 的进程创建一块共享内存，其余进程连接到这块共享内存，从而实现跨进程的数据共享。
        """
        if self.world_size > 1:
            if rank == 0:
                # 进程 rank == 0（主进程）创建一块大小为 1MB（2^20 = 1048576 字节）的共享内存，名字是 "nanovllm"
                self.shm = SharedMemory(name="nanovllm", create=True, size=2**20)
                # 创建之后调用 dist.barrier()，等待其他进程完成自己的部分。
                dist.barrier()
            else:
                # 其他进程（非 rank 0）先等待 barrier（确保主进程已经创建好了共享内存）。
                dist.barrier()
                # 然后连接到主进程创建好的共享内存（通过名称 "nanovllm"）
                self.shm = SharedMemory(name="nanovllm")
                # 最后调用 self.loop()，这个函数可能是该进程要执行的主要工作循环（比如处理请求、推理等）
                self.loop()

    def exit(self):
        """
        清理共享内存资源，并确保所有进程同步后才解除共享内存链接。
        """
        if self.world_size > 1:
            self.shm.close()
            dist.barrier()
            if self.rank == 0:
                self.shm.unlink()
        # dist.destroy_process_group()

    def loop(self):
        """
        这是一个无限循环函数，用于在非主进程中持续监听来自共享内存（Shared Memory）的消息，并根据消息内容调用相应的函数。
        """
        while True: # 进入一个无限循环，意味着该进程会一直运行下去，直到被显式终止或收到退出指令。
            method_name, args = self.read_shm() # 从共享内存中读取一条消息
            self.call(method_name, *args)
            if method_name == "exit":
                break

    def read_shm(self):
        """
        ✅ 功能说明：
        只有非主进程（rank != 0）调用。
        使用 event.wait() 阻塞等待主进程写入新任务。
        从共享内存中读取一个打包好的 (method_name, *args)。
        返回方法名和参数，供后续调用。
        🧠 设计思想：
        使用共享内存缓冲区前4字节保存数据长度，避免粘包问题；
        pickle 序列化/反序列化用于传输任意 Python 对象；
        event.wait() 和 .clear() 实现同步机制，防止数据竞争。
        """
        
        assert self.world_size > 1 and self.rank
        self.event.wait() # 等待主进程写入数据
        n = int.from_bytes(self.shm.buf[0:4], "little") # 读取数据长度
        method_name, *args = pickle.loads(self.shm.buf[4:n+4]) # 读取方法名和参数
        self.event.clear()
        return method_name, args

    def write_shm(self, method_name, *args):
        """
        ✅ 功能说明：
        只有主进程（rank == 0）调用。
        将 (method_name, *args) 打包成二进制数据；
        写入共享内存；
        最后通过 event.set() 唤醒所有等待中的子进程。
        🧠 设计思想：
        主动向共享内存写入任务；
        支持广播给多个子进程（比如多个 worker）；
        使用 assert 确保数据不会超出共享内存大小限制。
        """
        
        assert self.world_size > 1 and not self.rank
        data = pickle.dumps([method_name, *args])
        n = len(data)
        assert n + 4 <= self.shm.size
        self.shm.buf[0:4] = n.to_bytes(4, "little")
        self.shm.buf[4:n+4] = data
        for event in self.event:
            event.set()

    def call(self, method_name, *args):
        """
        ✅ 功能说明：
        如果是主进程，则将方法名和参数写入共享内存；
        不论是否主进程，都会调用本地对应的方法；
        子进程会通过 loop() 循环监听并执行相同的方法。
        🧠 设计思想：
        主进程调用该函数时，相当于“发送指令”给子进程；
        子进程接收到指令后，也调用同样的方法，形成“远程过程调用”；
        所有进程都共享同一套方法定义，因此可以通过名字调用。
        """
        
        if self.world_size > 1 and self.rank == 0:
            self.write_shm(method_name, *args)
        method = getattr(self, method_name, None)
        assert callable(method)
        return method(*args)

    def allocate_kv_cache(self, gpu_memory_utilization):
        config = self.config
        hf_config = config.hf_config
        total, used, _ = get_gpu_memory()
        free = total * gpu_memory_utilization - used # 计算出可用于 KV 缓存的空闲内存 free，它是总内存乘以利用率减去已用内存。
        num_kv_heads = hf_config.num_key_value_heads // dist.get_world_size() # 计算每个进程的 KV 头数：num_kv_heads 是通过将总的 KV 头数除以分布式环境中的进程数量得到的。这是因为如果使用了张量并行化，则每个进程只需要处理一部分头。
        block_bytes = 2 * hf_config.num_hidden_layers * self.block_size * num_kv_heads * hf_config.head_dim * hf_config.torch_dtype.itemsize # 计算每个块的字节数：block_bytes 计算了每个 KV 缓存块所需的字节数。这里考虑了模型的层数、块大小、头的数量、头维度以及数据类型（例如 float32 或 float16）的大小。
        config.num_kvcache_blocks = int(free) // block_bytes # 确定可以分配多少个 KV 缓存块：基于计算出的空闲内存和每个块的大小，计算出可以在当前 GPU 上分配的最大 KV 缓存块数 num_kvcache_blocks。
        """
        创建 KV 缓存张量：
        使用 torch.zeros 创建一个形状为 (2, num_hidden_layers, num_kvcache_blocks, block_size, num_kv_heads, head_dim) 的零张量，用于存储 KV 缓存。
        第一维大小为2是因为我们需要分别存储键（K）和值（V）缓存。
        """
        self.kv_cache = torch.zeros(2, hf_config.num_hidden_layers, config.num_kvcache_blocks, self.block_size, num_kv_heads, hf_config.head_dim)
        layer_id = 0
        """
        关联模块与缓存：
        遍历模型的所有模块，对于具有 k_cache 和 v_cache 属性的模块，将其指向新分配的 KV 缓存张量的相应部分。
        这样做的目的是确保每个层都能访问到其对应的 KV 缓存，从而支持高效的前向传播过程。
        """
        for module in self.model.modules():
            if hasattr(module, "k_cache") and hasattr(module, "v_cache"):
                module.k_cache = self.kv_cache[0, layer_id]
                module.v_cache = self.kv_cache[1, layer_id]
                layer_id += 1

    def prepare_block_tables(self, seqs: list[Sequence]):
        max_len = max(len(seq.block_table) for seq in seqs)
        block_tables = [
            seq.block_table + [-1] * (max_len - len(seq.block_table))
            for seq in seqs
        ]
        block_tables = torch.tensor(block_tables, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        return block_tables

    def prepare_prefill(self, seqs: list[Sequence]):
        input_ids = []
        positions = []
        cu_seqlens_q = [0]
        cu_seqlens_k = [0]
        max_seqlen_q = 0
        max_seqlen_k = 0
        slot_mapping = []
        context_lens = None
        block_tables = None
        for seq in seqs:
            seqlen = len(seq)
            input_ids.extend(seq[seq.num_cached_tokens:])
            positions.extend(list(range(seq.num_cached_tokens, seqlen)))
            seqlen_q = seqlen - seq.num_cached_tokens
            seqlen_k = seqlen
            cu_seqlens_q.append(cu_seqlens_q[-1] + seqlen_q)
            cu_seqlens_k.append(cu_seqlens_k[-1] + seqlen_k)
            max_seqlen_q = max(seqlen_q, max_seqlen_q)
            max_seqlen_k = max(seqlen_k, max_seqlen_k)
            for i in range(seq.num_cached_blocks, seq.num_blocks):
                start = seq.block_table[i] * self.block_size
                if i != seq.num_blocks - 1:
                    end = start + self.block_size
                else:
                    end = start + seq.last_block_num_tokens 
                slot_mapping.extend(list(range(start, end)))
        assert len(input_ids) == len(slot_mapping)
        assert len(input_ids) == cu_seqlens_q[-1]
        if cu_seqlens_k[-1] > cu_seqlens_q[-1]:    # prefix cache
            context_lens = torch.tensor([len(seq) for seq in seqs], dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
            block_tables = self.prepare_block_tables(seqs)
        input_ids = torch.tensor(input_ids, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        positions = torch.tensor(positions, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        cu_seqlens_q = torch.tensor(cu_seqlens_q, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        cu_seqlens_k = torch.tensor(cu_seqlens_k, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        slot_mapping = torch.tensor(slot_mapping, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        set_context(True, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, slot_mapping, context_lens, block_tables)
        return input_ids, positions

    def prepare_decode(self, seqs: list[Sequence]):
        input_ids = []
        positions = []
        slot_mapping = []
        context_lens = []
        for seq in seqs:
            input_ids.append(seq.last_token)
            positions.append(len(seq))
            context_lens.append(len(seq))
            slot_mapping.append(seq.block_table[-1] * self.block_size + seq.last_block_num_tokens  - 1)
        input_ids = torch.tensor(input_ids, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        positions = torch.tensor(positions, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        slot_mapping = torch.tensor(slot_mapping, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        context_lens = torch.tensor(context_lens, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        block_tables = self.prepare_block_tables(seqs)
        set_context(False, slot_mapping=slot_mapping, context_lens=context_lens, block_tables=block_tables)
        return input_ids, positions

    def prepare_sample(self, seqs: list[Sequence]):
        temperatures = []
        for seq in seqs:
            temperatures.append(seq.temperature)
        temperatures = torch.tensor(temperatures, dtype=torch.float32, pin_memory=True).cuda(non_blocking=True)
        return temperatures

    @torch.inference_mode() # 相比于 torch.no_grad()，inference_mode() 更加高效，因为它还会优化内存使用和计算图构建。
    def run_model(self, input_ids: torch.Tensor, positions: torch.Tensor, is_prefill):
        """
        args:
            input_ids: 当前 batch 的 token IDs，形状为 (batch_size, seq_len)。
            positions: token 的位置索引，用于位置编码。
            is_prefill: 布尔值，表示是否是预填充阶段（prefill）。如果是，则处理的是长序列的首次推理。

        +----------------------------+
        |         run_model()         |
        +----------------------------+
                    ↓
             是否使用 CUDA Graph?
               ┌───────────────┐
               │               │
           是 prefilled?    batch > 512?
               │               │
               └──────┬────────┘
                      ↓
             +-------------------+
             | 直接前向传播推理     |
             | model(input_ids, ...)|
             +-------------------+
                      ↓
             返回 logits 给 sampler
        
                      ↓
              +------------------+
              | 使用 CUDA Graph   |
              | 加载对应 batch 图 |
              | 拷贝输入进 buffer |
              | replay()          |
              +------------------+
                      ↓
             返回 logits 给 sampler
        """
        # 判断是否使用 eager 模式或直接推理
        if is_prefill or self.enforce_eager or input_ids.size(0) > 512:
            """
            is_prefill == True：当前是“预填充”阶段（即处理 prompt 阶段），通常序列较长，不适合使用 CUDA 图加速。
            self.enforce_eager == True：用户强制要求不使用 CUDA 图。
            input_ids.size(0) > 512：批量太大，超出了预先捕获的 CUDA 图支持的最大 batch size。
            直接调用模型的 forward 方法进行前向传播；
            然后通过 compute_logits(...) 获取最终的输出 logits；
            这是最简单、最通用的执行方式。
            """
            return self.model.compute_logits(self.model(input_ids, positions))
        else:
            bs = input_ids.size(0) # 获取 batch size
            context = get_context() # 获取上下文信息
            self.reset_graph_vars() # 重置图变量缓存, 清除之前缓存的输入数据，避免脏数据干扰。通常会重置 input_ids, positions, outputs 等张量。
            '''
            作用：
                从预先捕获好的多个 CUDA 图中选择一个能容纳当前 batch 大小的图。
                self.graph_bs 是一个排序好的列表，比如 [1, 4, 16, 64, 256, 512]；
                使用 next(...) 找到第一个大于等于当前 bs 的项作为匹配项。
            '''
            graph = self.graphs[next(x for x in self.graph_bs if x >= bs)] # 选择合适 batch size 的 CUDA 图
            graph_vars = self.graph_vars # 包含所有需要提前分配的张量缓冲区
            graph_vars["input_ids"][:bs] = input_ids
            graph_vars["positions"][:bs] = positions
            graph_vars["slot_mapping"][:bs] = context.slot_mapping
            graph_vars["context_lens"][:bs] = context.context_lens
            graph_vars["block_tables"][:bs, :context.block_tables.size(1)] = context.block_tables
            graph.replay() # 执行之前捕获的 CUDA 图，快速完成整个前向传播过程。节省了每次调用模型时重新构建计算图的开销；特别适合固定输入 shape 的 decode 阶段；性能提升可达数倍。
            return self.model.compute_logits(graph_vars["outputs"][:bs])

    def reset_graph_vars(self):
        graph_vars = self.graph_vars
        graph_vars["input_ids"].zero_()
        graph_vars["positions"].zero_()
        graph_vars["slot_mapping"].zero_()
        graph_vars["context_lens"].zero_()
        graph_vars["block_tables"].zero_()
    
    def run(self, seqs: list[Sequence], is_prefill: bool) -> list[int]:
        input_ids, positions = self.prepare_prefill(seqs) if is_prefill else self.prepare_decode(seqs)
        temperatures = self.prepare_sample(seqs)
        logits = self.run_model(input_ids, positions, is_prefill)
        token_ids = self.sampler(logits, temperatures).tolist() if self.rank == 0 else None
        reset_context()
        return token_ids

    @torch.inference_mode()
    def capture_cudagraph(self):
        '''
        负责在模型推理阶段预先捕获 CUDA Graph（CUDA 计算图），以便在后续的 decode 阶段进行高性能、低延迟的图回放（graph replay）。
        '''
        get_rng_state = torch.cuda.get_rng_state
        set_rng_state = torch.cuda.set_rng_state
        rng_state = torch.cuda.get_rng_state()
        torch.cuda.get_rng_state = lambda: rng_state
        torch.cuda.set_rng_state = lambda _: None
        '''
        以上代码:
            冻结当前 GPU 的随机数生成器状态（RNG），避免 CUDA 图中出现不可预测的操作（如 dropout、随机初始化等）导致图无法复现。
        '''
        config = self.config
        hf_config = config.hf_config
        max_bs = min(self.config.max_num_seqs, 512)
        max_num_blocks = (config.max_model_len + self.block_size - 1) // self.block_size
        '''
        配置信息提取：
            hf_config: HuggingFace 模型配置；
            max_bs: 最大 batch size，取 max_num_seqs 和 512 中较小者；
            max_num_blocks: 根据最大序列长度和块大小计算出的最大 KV 缓存块数量（用于 PagedAttention）。
        '''
        input_ids = torch.zeros(max_bs, dtype=torch.int64)
        positions = torch.zeros(max_bs, dtype=torch.int64)
        slot_mapping = torch.zeros(max_bs, dtype=torch.int32)
        context_lens = torch.zeros(max_bs, dtype=torch.int32)
        block_tables = torch.zeros(max_bs, max_num_blocks, dtype=torch.int32)
        outputs = torch.zeros(max_bs, hf_config.hidden_size)
        self.graph_bs = [1, 2, 4, 8] + list(range(16, max_bs + 1, 16)) # 定义一组支持的 batch size（例如 [1, 2, 4, 8, 16, 32, ..., 512]）, 用于后续匹配合适的 CUDA Graph
        self.graphs = {} # self.graphs: 存储每个 batch size 对应的 CUDA Graph
        self.graph_pool = None # self.graph_pool: 图池，用于重用内存分配资源。

        for bs in reversed(self.graph_bs):
            '''
            ✅ 循环逻辑详解：
                按从大到小顺序遍历所有支持的 batch size；
                创建新的 CUDAGraph() 实例；
                调用 set_context(...) 设置当前上下文（模拟请求的 slot mapping、block table 等）；
                第一次前向传播（warmup）：
                执行一次模型前向传播，预热 GPU，防止首次运行引入额外开销；
                第二次前向传播（图捕获）：
                使用 with torch.cuda.graph(graph, pool): 捕获整个计算过程；
                这次不会立即执行，而是记录成图；
                首次捕获后保存图池；
                保存当前 batch size 对应的图到 self.graphs 字典中；
                同步设备，确保图构建完成；
                调用 reset_context() 清除临时上下文
            '''
            graph = torch.cuda.CUDAGraph()
            set_context(False, slot_mapping=slot_mapping[:bs], context_lens=context_lens[:bs], block_tables=block_tables[:bs])
            outputs[:bs] = self.model(input_ids[:bs], positions[:bs])    # warmup
            with torch.cuda.graph(graph, self.graph_pool):
                outputs[:bs] = self.model(input_ids[:bs], positions[:bs])    # capture
            if self.graph_pool is None:
                self.graph_pool = graph.pool()
            self.graphs[bs] = graph
            torch.cuda.synchronize()
            reset_context()

        self.graph_vars = dict(
            input_ids=input_ids,
            positions=positions,
            slot_mapping=slot_mapping,
            context_lens=context_lens,
            block_tables=block_tables,
            outputs=outputs,
        )

        torch.cuda.get_rng_state = get_rng_state
        torch.cuda.set_rng_state = set_rng_state
