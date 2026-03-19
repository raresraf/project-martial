// Problem: Execute a series of increment or decrement operations and find the final value of a variable.
// This is a competitive programming problem involving basic string parsing and counting.
// The operations are of the form "++X", "X++", "--X", or "X--".
// The key insight is that all increment operations contain a '+' and all decrement operations contain a '-'.
// A simple way to distinguish them is by checking the character at index 1 of the operation string.
// Time Complexity: O(N), where N is the number of operations, as we process each operation once.
// Space Complexity: O(1), aside from storing the input string for each operation.
#include <cstdio>
#include <iostream>
using namespace std;

int main() {
  int numOperations;
  scanf("%d
", &numOperations);
  string temp = "";
  // 'output' will store the final value after all operations.
  int output = 0;
  // Loop through all the given operations.
  while (--numOperations >= 0) {
    getline(cin, temp);
    // The character at index 1 uniquely determines the operation type.
    // If it's '+', it's an increment operation.
    if (temp[1] == '+') {
      ++output;
    } else {
      // Otherwise, it's a decrement operation.
      --output;
    }
  }
  printf("%d
", output);
  return 0;
}
