// code by Nikhil Nagrale
// nikhilnagrale2 on EveryPlatform else nknagrale
/**
 * @file 282A.CPP
 * @brief Solution to the "Bit++" competitive programming problem.
 * @details This program calculates the final value of a variable 'X' which starts at 0
 * and is modified by a series of increment ("++X", "X++") and decrement ("--X", "X--") operations.
 *
 * @section algorithm Algorithm
 * The algorithm iterates through each operation and updates a counter:
 * 1. Read the total number of operations, `n`.
 * 2. Initialize a counter, `ans`, to 0. This variable represents the value of 'X'.
 * 3. Loop `n` times to process each operation.
 * 4. In each iteration, read the operation as a string.
 * 5. If the string matches an increment operation, increment `ans`.
 * 6. Otherwise, it must be a decrement operation, so decrement `ans`.
 * 7. After the loop completes, print the final value of the counter.
 *
 * @section complexity Complexity Analysis
 * - Time Complexity: O(N), where N is the number of operations. String comparisons are effectively constant time as the input strings are short.
 * - Space Complexity: O(1), as memory usage does not scale with the input.
 */
#include <bits/stdc++.h>
using namespace std;

int main() {
  int n; // Stores the number of operations.
  cin >> n;
  int ans = 0; // 'ans' represents the value of X, initialized to 0.

  // Loop n times, processing one operation in each iteration.
  while (n--) {
    string a; // Variable to store the operation string (e.g., "++X").
    cin >> a;
    
    // Check if the read operation is an increment.
    if (a == "++X" || a == "X++")
      ans++;
    else // Otherwise, the operation must be a decrement.
      ans--;
  }
  
  // Print the final value of X after all operations are performed.
  cout << ans << endl;
  return 0;
}