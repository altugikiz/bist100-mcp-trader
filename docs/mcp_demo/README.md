# MCP Demo Screenshots

This folder contains five screenshots captured from Claude Desktop demonstrating the `bist-analytics` MCP server in action.

## What the demo shows

An ASELS.IS analysis query was submitted to Claude Desktop. Claude automatically called all six MCP tools exposed by `src/mcp_server.py` in sequence — fetching price history, computing technical indicators, locating support/resistance levels, retrieving fundamental metrics, scoring fundamental quality, and summarising backtest performance — then synthesised the results into a single investment analysis response.

## Reading order

Read the screenshots from **part1 → part5** in order. Each image captures a consecutive portion of the Claude Desktop conversation window:

| File | Content |
|---|---|
| `mcp_demo_part1.png` | User query and start of tool calls |
| `mcp_demo_part2.png` | Continued tool calls (indicators, support/resistance) |
| `mcp_demo_part3.png` | Fundamental metrics and scoring tool calls |
| `mcp_demo_part4.png` | Backtest summary tool call and raw results |
| `mcp_demo_part5.png` | Claude's synthesised analysis response |
