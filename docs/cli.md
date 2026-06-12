# CLI Reference

## Usage

```bash
cognitive-engine [OPTIONS] FILE
```

## Arguments

| Argument | Description |
|----------|-------------|
| `FILE` | Path to a plain text file to analyze (required) |

## Options

| Flag | Default | Description |
|------|---------|-------------|
| `--output PATH` | stdout | Write JSON output to a file instead of printing |
| `--config PATH` | built-in defaults | Path to a JSON priors config file |
| `--mode MODE` | `argument` | Reasoning mode: `causal`, `conditional`, `argument`, or `analogy` |
| `--chunk-size N` | 512 | Maximum tokens per sliding window chunk |
| `--chunk-overlap N` | 128 | Token overlap between adjacent chunks |

## Examples

### Basic analysis

```bash
cognitive-engine demo/complicated.txt
```

Prints the full JSON graph to stdout.

### Save output to file

```bash
cognitive-engine demo/complicated.txt --output result.json
```

### Custom chunking

```bash
cognitive-engine demo/complicated.txt --chunk-size 256 --chunk-overlap 64
```

Larger overlaps reduce the chance of splitting a proposition across chunk boundaries, at the cost of more processing.

### Reasoning mode filter

```bash
cognitive-engine demo/complicated.txt --mode causal
```

Filters the output graph to only edges relevant to causal reasoning (INFERS, SUPPORTS).

### Custom priors

```bash
cognitive-engine demo/complicated.txt --config my_priors.json
```

The JSON file should match the schema defined in [Configuration](configuration.md).
