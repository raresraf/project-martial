/**
 * @file 282A.CPP
 * @brief Solution for the "Bit++" problem.
 * @details This program calculates the final value of a variable 'x' after a series
 * of increment and decrement operations. The operations are given as strings.
 *
 * @section algorithm Algorithm
 * 1. An integer `n` is read, representing the total number of operations.
 * 2. A variable `ans` is initialized to 0, which will hold the final value.
 * 3. The program loops `n` times. In each iteration:
 *    a. An operation string (e.g., "++X", "X--") is read.
 *    b. The string is checked. If it matches "++X" or "X++", the `ans` variable is incremented.
 *    c. Otherwise, it's assumed to be a decrement operation, and `ans` is decremented.
 * 4. After all operations are processed, the final value of `ans` is printed.
 *
 * @section complexity Complexity Analysis
 * - Time Complexity: O(n), where `n` is the number of operations, because the loop runs `n` times.
 * - Space Complexity: O(1), as the space required for variables is constant, and the string
 *   storage is reused in each iteration of the loop.
 */
// code by Nikhil Nagrale
// nikhilnagrale2 on EveryPlatform else nknagrale
#include <bits/stdc++.h>
using namespace std;

/**
 * @brief Main function to process the Bit++ operations.
 * @return 0 on successful execution.
 */
int main() {
  int n;
  cin >> n;
  int ans = 0; // This variable will store the final value of X.
  
  // Loop 'n' times to process each operation.
  while (n--) {
    string a;
    cin >> a;
    
    // Check if the operation is an increment.
    if (a == "++X" || a == "X++") {
      ans++;
    } else { // Otherwise, it's a decrement.
      ans--;
    }
  }
  
  // Print the final result.
  cout << ans << endl;
  return 0;
}