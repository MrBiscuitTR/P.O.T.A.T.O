---
name: "Graph Generation Tool"
description: "Generate charts and graphs from data using potatool_generate_graph"
alwaysApply: true
---

# Graph Generation Tool: potatool_generate_graph

Use this tool when the user asks for a chart, graph, visualization, or when data clearly benefits from visual representation.

## Tool: potatool_generate_graph

### Parameters

- **data** (required, string): JSON array of data objects with consistent keys. Example: `[{"month":"Jan","value":100},{"month":"Feb","value":120}]`
- **graph_type** (required, string): One of: line, bar, scatter, pie, histogram, box, area, heatmap, radar, donut, funnel, waterfall, step, errorbar
- **title** (required, string): Descriptive graph title
- **x_key** (optional, string): Key name for X-axis values. Auto-detected from first key if omitted.
- **y_key** (optional, string): Key name for Y-axis values. Auto-detected from second key if omitted.
- **x_label** (optional, string): X-axis label
- **y_label** (optional, string): Y-axis label

### When to Use

- User explicitly asks for a chart, graph, plot, or visualization
- User provides data (table, list, numbers) and asks to visualize it
- When comparing values, showing trends, or illustrating proportions

### Graph Type Recommendations

- **line**: Trends over time, continuous data
- **bar**: Comparing categories, discrete values
- **scatter**: Correlation between two variables
- **pie/donut**: Proportions of a whole (use with <8 categories)
- **histogram**: Distribution of a single variable
- **box**: Statistical distribution summary
- **area**: Cumulative trends, volume over time
- **radar**: Multi-dimensional comparison
- **funnel**: Sequential stages with drop-off
- **waterfall**: Cumulative positive/negative changes

### Rules

1. Always provide a descriptive title that explains what the graph shows
2. Label axes clearly using x_label and y_label
3. Choose the most appropriate graph type for the data
4. Keep data clean — use consistent keys across all objects
5. For pie/donut charts, limit to 8 or fewer categories for readability
6. After the tool returns, include the markdown image in your response so the user sees it
7. The tool returns a `markdown` field — include it in your response verbatim
