import argparse
from src.benchmarks import RecallBenchmark
from src.benchmarks import DissmatchBenchmark
from src.dataset import load_config

def run_benchmark(config_path, benchmark_type, batch_size=None, use_random_candidates=False, num_candidates=40):
    if benchmark_type == "recall":
        benchmark = RecallBenchmark(config_path)
    elif benchmark_type == "dissmatch":
        benchmark = DissmatchBenchmark(
            config_path,
            batch_size=batch_size,
            use_random_candidates=use_random_candidates,
            num_candidates=num_candidates
        )
    else:
        raise ValueError("Unknown benchmark type. Use 'recall' or 'dissmatch'.")

    benchmark.run()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run benchmarks.")
    parser.add_argument("benchmark_type", choices=["recall", "dissmatch"], help="Type of benchmark to run.")
    parser.add_argument("config_path", type=str, help="Path to the configuration file.")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for the dataloader (only for dissmatch).")
    parser.add_argument("--use_random_candidates", default=False, help="Use random candidates for the dataset (only for dissmatch).")
    parser.add_argument("--num_candidates", type=int, default=40, help="Number of random candidates to use (only for dissmatch).")

    args = parser.parse_args()
    run_benchmark(
        args.config_path,
        args.benchmark_type,
        args.batch_size if args.benchmark_type == "dissmatch" else None,
        args.use_random_candidates if args.benchmark_type == "dissmatch" else False,
        args.num_candidates if args.benchmark_type == "dissmatch" else 40
    )