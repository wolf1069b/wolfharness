---
name: test-with-args
description: A skill that accepts arguments for testing parameter passing and input validation across protocols
license: Apache-2.0
compatibility: 1.0.0
allowed-tools: bash, read, grep
metadata:
  category: testing
  complexity: intermediate
---

# test-with-args

A skill that accepts arguments for testing parameter passing

## License
Apache-2.0

## Compatibility
1.0.0

## Allowed Tools
bash, read, grep

## Instructions

This skill demonstrates argument handling and parameter validation.

When invoked with arguments, process them appropriately:
- Echo back the provided arguments
- Validate argument format if specified
- Return structured response with processed input

### Usage Examples

With single argument:
```bash
wolfharness skill test-with-args "my test input"
```

With multiple arguments:
```bash
wolfharness skill test-with-args arg1 arg2 arg3
```

Expected output:
- Confirmation of received arguments
- Processed result based on input
- Error message for invalid input

### Testing Scenarios

1. Argument Passing: Verify arguments are correctly passed through:
   - ACP: command with input field
   - AG-UI: tool with arguments parameter
   - OpenCode: command with args list

2. Input Validation: Test validation behavior with:
   - Empty arguments
   - Special characters
   - Unicode input
   - Long strings

3. Protocol-Specific Format Verification:
   - ACP AvailableCommand has correct input spec
   - AG-UI Tool has correct parameters schema
   - OpenCode command has proper usage hint
