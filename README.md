# DecisionTrees

Currently:
- Includes tokenizer
- Includes kfold cross validation
- Includes pre and post pruning
- Supports classification only
- Supports discrete inputs only
- Supports determinstic relations only
- Supports node splitting using equality and inequality relations (inequality or binary splitting leads to simpler trees)

To do:
- Include pre-pruning for binary splitting
- Support non-deterministic relations (easy to implement, deal with situations where attr_list is empty but entropy is not zero)
- Improve tree visualization
- Clean code
- Add better documentation
- Support continuous inputs
- Support regression
