/**
 * @file 282A.CPP
 * @brief Solves the "Bit++" problem from Codeforces.
 * @details The problem involves a variable X, initially 0, and a sequence of operations
 * that either increment or decrement X. The goal is to find the final value of X.
 *
 * @algorithm
 * The solution initializes a counter variable to zero. It then reads the number of
 * operations to be performed. In a loop that runs for the given number of times,
 * it reads each operation as a string. It uses string comparison to check if the
 * operation is an increment ("++X" or "X++") or a decrement ("--X" or "X--")
 * and updates the counter accordingly. Finally, it prints the counter's final value.
 *
 * @complexity
 * Time Complexity: O(N), where N is the number of operations, because the loop runs N times
 * and the string comparisons inside it take constant time (since strings are of fixed, short length).
 * Space Complexity: O(1), as it only uses a few variables to store the counter and the current operation string.
 */
#include <cstring>
#include <iostream>

using namespace std;
int main() {
  /**
   * @variable t: The total number of operations to be performed.
   * @variable x: The variable whose value is modified, initialized to 0.
   */
  int t, x = 0;
  /**
   * @variable str: A character array to store the operation string (e.g., "X++").
   * Size 4 is sufficient for a 3-character string plus the null terminator.
   */
  char str[4];
  
  // Read the number of operations.
  cin >> t;
  
  /**
   * @block
   * @brief This loop processes each of the `t` operations.
   * @invariant At the beginning of each iteration, `x` holds the accumulated value from all previous operations.
   */
  for (; t > 0; t--) {
    // Read the next operation string.
    cin >> str;
    
    // Pre-condition: Check if the operation is an increment.
    if ((strcmp(str, "X++") == 0) || (strcmp(str, "++X") == 0))
      x = x + 1;
      
    // Pre-condition: Check if the operation is a decrement.
    if ((strcmp(str, "X--") == 0) || (strcmp(str, "--X") == 0))
      x = x - 1;
  }
  
  // Output the final value of x after all operations.
  cout << x << endl;
  return 0;
}
