/**
 * @file 282A.CPP
 * @brief Solution for the "Bit++" problem from Codeforces.
 * @details This program calculates the final value of a variable 'X' after a series
 * of increment and decrement operations. The operations are given as strings like
 * "++X", "X++", "--X", or "X--".
 *
 * @algorithm Simple Counter
 * The program initializes a counter to zero. It then iterates through each operation.
 * If the operation string contains a '+', the counter is incremented. Otherwise,
 * it's assumed to contain a '-' and the counter is decremented. The logic relies
 * on the fact that any increment operation ("++X" or "X++") will have a '+' at
 * either the first or third character position.
 *
 * @complexity
 * - Time: O(N), where N is the number of statements. The program reads and processes
 *   each of the N strings once.
 * - Space: O(1), as it only stores a few variables and the current operation string.
 */

#include <bits/stdc++.h> // Includes all standard C++ libraries.
using namespace std;

// Common competitive programming macros, though not used in this specific solution.
#define f first         // Alias for the first element of a pair.
#define s second        // Alias for the second element of a pair.
#define mp make_pair    // Alias for std::make_pair.
#define pb push_back    // Alias for std::vector::push_back.

// Typedef for long long integers for convenience.
typedef long long ll;

int main() {
  // The following line is often used in competitive programming for local testing
  // to read input from a file named "input.txt" instead of standard input.
  // freopen("input.txt", "r", stdin);
  
  // Optimizes C++ stream I/O by decoupling from C-style I/O.
  ios_base::sync_with_stdio(false);

  // n: the number of statements to be executed.
  int n;
  cin >> n;

  // `count` will store the final value of the variable X, initialized to 0.
  int count = 0;

  /**
   * @brief Process each statement.
   * The loop runs `n` times, once for each statement.
   */
  // Invariant: `count` holds the value of X after processing `n` minus the current `n` statements.
  while (n--) {
    string xs; // `xs` will hold the operation string (e.g., "++X").
    cin >> xs;

    /**
     * @brief Check for increment or decrement operation.
     * An increment operation like "++X" or "X++" will always have a '+'
     * at index 0 or index 2. Any other operation must be a decrement.
     */
    // Pre-condition: Check if the statement is an increment.
    if (xs[0] == '+' || xs[2] == '+')
      ++count;
    else
      --count;
  }

  // Output the final value of the counter.
  cout << count << endl;
  
  return 0; // Indicate successful execution.
}
