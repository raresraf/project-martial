/**
 * @file 282A.CPP
 * @brief Solution to a competitive programming problem (Codeforces 282A - "Bit++").
 *
 * This program simulates a programming language with a single variable `X`
 * and two types of operations: increment (`++X` or `X++`) and decrement (`--X` or `X--`).
 * The goal is to determine the final value of `X` after a sequence of `n` operations,
 * starting with an initial value of 0.
 *
 * Algorithm: Expression parsing and simple arithmetic.
 * The program reads `n` operation strings. For each string, it inspects the second
 * character (`x[1]`) to determine if the operation is an increment ('+') or a
 * decrement ('-'). A counter (`count`) is updated accordingly.
 *
 * Time Complexity: O(N), where N is the number of operations.
 * Each operation involves reading a small string and performing constant-time checks
 * and arithmetic.
 * Space Complexity: O(1) (excluding input string storage), as only a few integer variables
 * are used. The character array `x` has a fixed maximum size.
 */
#include <bits/stdc++.h> // Includes various standard libraries for C++.
#define pi 2.0 * acos(0.0) // Defines pi, not used in this solution.
#define MAX 100005         // Defines a maximum size for character array, not directly 'N' in complexity.
typedef long long ll;      // Type alias for long long, not used for main variables.
using namespace std;

/**
 * @brief Main function: Entry point of the program.
 *
 * Reads the number of operations and then processes each operation to compute
 * the final value of a counter.
 */
int main() {
  int i, j, k, n, count = 0; // 'n': number of operations, 'count': tracks the variable's value.
                           // 'i', 'j', 'k' are declared but unused.
  char x[MAX];             // Character array to store each operation string.
  // Pre-condition: Read 'n', the total number of operations.
  cin >> n;
  // Block Logic: Iterate 'n' times, once for each operation.
  // Invariant: 'i' represents the current operation index.
  for (i = 0; i < n; i++) {
    // Pre-condition: Read the operation string (e.g., "++X", "X--").
    cin >> x;
    // Block Logic: Check the second character of the string to determine the operation.
    // Functional Utility: This works because the increment/decrement operators are always in the middle position.
    if (x[1] == '+')
      count = count + 1; // Increment the counter.
    else if (x[1] == '-')
      count = count - 1; // Decrement the counter.
  }
  // Output the final computed value of the counter.
  cout << count << endl; // Use endl for flushing output.
  return 0; // Indicate successful execution.
}
