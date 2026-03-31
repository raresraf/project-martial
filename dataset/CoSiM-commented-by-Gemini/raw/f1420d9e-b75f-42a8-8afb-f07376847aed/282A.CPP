/**
 * @file 282A.CPP
 * @brief Simulates increment/decrement operations based on string inputs.
 *
 * This program models a simple computational problem where an initial variable,
 * `ans`, is modified over a series of operations. It reads an integer `t` which
 * specifies the number of operations. In a loop that runs `t` times, it reads a
 * string representing an operation. The operation is either an increment or a
 * decrement, determined by the characters in the string (e.g., "++X", "X--").
 * The final value of `ans` is printed after all operations are completed.
 *
 * @author f1420d9e-b75f-42a8-8afb-f07376847aed
 *
 * @algorithm String-based Simulation
 * @complexity
 * Time: O(T), where T is the number of operations, due to the single loop processing each string.
 * Space: O(1), as the memory used is constant regardless of the input size.
 */
#include <bits/stdc++.h>
using namespace std;

int main() {
  // `t` stores the number of operations to be performed.
  int t;
  cin >> t;
  // `ans` is the variable that will be modified, initialized to 0.
  int ans = 0;

  /**
   * @brief This loop executes `t` times, once for each operation.
   * Pre-condition: `ans` holds the accumulated value from prior iterations.
   * Invariant: `ans` is updated within the loop based on the string read.
   */
  for (int i = 0; i < t; i++) {
    // The line `// int ans=0;` is commented out and has no effect.
    string s;
    cin >> s;

    /**
     * @brief These conditional statements determine the operation type.
     * The logic checks characters at specific positions in the string to decide
     * whether to increment or decrement the `ans` variable.
     * This approach correctly handles formats like "++X", "X++", "--X", and "X--".
     */
    if (s[0] == '+') { // Handles prefix increment, e.g., "++X"
      ++ans;
    }
    if (s[0] == '-') { // Handles prefix decrement, e.g., "--X"
      --ans;
    }
    if (s[2] == '+') { // Handles postfix increment, e.g., "X++"
      ans++;
    }
    if (s[2] == '-') { // Handles postfix decrement, e.g., "X--"
      ans--;
    }
    // The line `// cout << ans << endl;` is commented out and has no effect.
  }
  // After the loop, the final value of `ans` is printed.
  cout << ans << endl;

  return 0;
}