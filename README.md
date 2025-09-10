# Adel-Lite: Automated Data Elements Linking - Lite

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/release/python-380/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**Adel-Lite** is a Python library for automated schema generation, data profiling, and relationship discovery for Pandas DataFrames. It helps you understand your data structure and relationships with minimal effort.

## Features

🔍 **Schema Generation**: Automatic structural schema detection  
📊 **Data Profiling**: Comprehensive statistics and semantic type inference  
🔗 **Relationship Mapping**: Primary/Foreign key detection using heuristics  
⚡ **Constraint Discovery**: Intra-row constraint detection (GT, EQ)  
📈 **Visualization**: Schema graphs with Graphviz  
📤 **Multi-format Export**: JSON, YAML, SQL DDL, Avro  
🛠️ **CLI Support**: Command-line interface for batch processing  

## Installation

```bash
pip install adel-lite
```
### Development installation
```bash
git clone https://github.com/Parthnuwal7/adel-lite.git
cd adel-lite
pip install -e .
```
## Quick Start
### Basic Usage

```python
import pandas as pd
from adel_lite import schema, profile, map_relationships, build_meta
```
### Load your data

```python
customers = pd.DataFrame({
'customer_id': ,
'name': ['Alice', 'Bob', 'Charlie'],
'email': ['alice@test.com', 'bob@test.com', 'charlie@test.com']
})

orders = pd.DataFrame({
'order_id': ,
'customer_id': ,
'amount': [100.0, 150.0, 75.0]
})

df_list = [customers, orders]
table_names = ['customers', 'orders']
```
### Generate comprehensive analysis

```python
schema_result = schema(df_list, table_names)
profile_result = profile(df_list, table_names)
relationships_result = map_relationships(df_list, table_names)
```
### Build final meta structure

```python
meta = build_meta(schema_result, profile_result, relationships_result)
print(json.dumps(meta, indent=2))
```
### Command Line usage

#### Analyze CSV files

```bash
adel-lite --input data/*.csv --output schema.json
```
#### Generate visualization

```bash
adel-lite --input *.csv --visualize --output schema.json
```
#### Export as SQL DDL

```bash
adel-lite --input data/*.csv --format ddl --output schema.sql
```
#### Skip constraint detection for faster processing

```bash
adel-lite --input *.csv --no-constraints --output schema.json
```
## Core Functions

### 1. Schema Generation

```python
from adel_lite import schema

Generate structural schema
result = schema(df_list, table_names)
```

**Returns:**
- Table names and column information
- Data types (pandas + high-level)
- Nullable flags and positions

### 2. Data Profiling
```python
from adel_lite import profile

Generate comprehensive profiles
result = profile(df_list, table_names)
```
**Returns:**
- Statistical summaries (min, max, mean, etc.)
- Uniqueness and null ratios
- Semantic type inference (id, datetime, categorical, etc.)
- Primary key candidates

### 3. Relationship Mapping
```python
from adel_lite import map_relationships

Detect relationships
result = map_relationships(df_list, table_names, fk_threshold=0.8)
```
**Returns:**
- Primary key detection
- Foreign key relationships with confidence scores
- Composite key candidates

### 4. Constraint Detection
```python
from adel_lite import detect_constraints

Find intra-row constraints
result = detect_constraints(df_list, table_names, threshold=0.95)
```
**Returns:**
- GT constraints: `A > B`
- EQ constraints: `A + B = C`
- Confidence scores

### 5. Visualization
```python
from adel_lite import visualize

Generate schema graph
path = visualize(schema_result, relationships_result, format='png')
```
### 6. Export

```python
from adel_lite import export_schema

Export to different formats
json_content = export_schema(meta, format='json')
yaml_content = export_schema(meta, format='yaml')
ddl_content = export_schema(meta, format='ddl')

```

## Example Output
```json
{
"metadata": {
"generated_at": "2025-09-10T12:42:00",
"generator": "adel-lite",
"version": "0.1.0"
},
"tables": [
{
"name": "customers",
"primary_key": "customer_id",
"fields": [
{
"name": "customer_id",
"dtype": "integer",
"semantic_type": "id",
"subtype": "primary",
"nullable": false
}
]
}
],
"relationships": [
{
"type": "foreign_key",
"foreign_table": "orders",
"foreign_column": "customer_id",
"referenced_table": "customers",
"referenced_column": "customer_id",
"confidence": 0.92
}
]
}
```

## Advanced Usage

### Custom Thresholds

Adjust detection thresholds

```python
relationships = map_relationships(
df_list, table_names,
fk_threshold=0.9, # Stricter FK detection
name_similarity_threshold=0.8
)

constraints = detect_constraints(
df_list, table_names,
threshold=0.98 # Very strict constraints
)

```

### Sampling and Inspection
```python
from adel_lite import sample

Get sample data for inspection
samples = sample(df_list, table_names, n=10, method='random')

Conditional sampling
samples = sample_by_condition(
df_list,
['age > 25', 'amount > 100'],
table_names
)

```

## Configuration

### CLI Configuration

Full configuration example
```bash 
adel-lite
--input data/*.csv
--output analysis.json
--format json
--visualize
--viz-format svg
--sample 5
--constraint-threshold 0.9
--fk-threshold 0.8
--verbose

```

### Logging

```python
import logging

#Enable debug logging
logging.getLogger('adel_lite').setLevel(logging.DEBUG)

```

## Performance Tips

1. **Skip constraints** for large datasets: `--no-constraints`
2. **Limit sampling** for inspection: `--sample 100`
3. **Use appropriate thresholds** based on data quality
4. **Process in batches** for very large datasets

## Requirements

- Python 3.8+
- pandas >= 1.3.0
- numpy >= 1.21.0
- pyyaml >= 6.0
- networkx >= 2.6
- matplotlib >= 3.5.0
- graphviz >= 0.20.0
- fuzzywuzzy >= 0.18.0

## Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature-name`
3. Make changes and add tests
4. Run tests: `pytest`
5. Submit a pull request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Roadmap

- [ ] Support for more data sources (databases, APIs)
- [ ] Advanced constraint types (LIKE patterns, regex)
- [ ] Machine learning-based relationship detection
- [ ] Interactive web interface
- [ ] Integration with data catalogs

## Support

- 📖 [Documentation](https://github.com/Parthnuwal7/adel-lite)
- 🐛 [Issue Tracker](https://github.com/Parthnuwal7/adel-lite)
- 💬 [Discussions](https://github.com/Parthnuwal7/adel-lite.git)

---

Made with ❤️ for the data community by Parth Nuwal