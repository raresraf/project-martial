/**
 * @file 282A.CPP
 * @brief Solution for the "Bit++" competitive programming problem.
 *
 * @details This program simulates a series of operations on a single variable,
 * initially zero. The operations are increments ("++X", "X++") and decrements
 * ("--X", "X--"). The final value of the variable after all operations is
 * calculated and printed.
 *
 * @section algorithm
 * Algorithm:
 * The program simulates the process directly.
 * 1. Initialize a counter variable to 0.
 * 2. Read the total number of operations, `n`.
 * 3. Loop `n` times:
 *    a. Read the operation as a string.
 *    b. If the string represents an increment operation, increment the counter.
 *    c. If the string represents a decrement operation, decrement the counter.
 * 4. After the loop, print the final value of the counter.
 *
 * @section complexity
 * Time Complexity: O(N), where N is the number of statements. The program
 * iterates through all `n` statements once, and the string comparisons inside
 * the loop take constant time.
 * Space Complexity: O(N), as the implementation stores all `n` operation strings
 * in a vector.
 */
#include <bits/stdc++.h>
using namespace std;

int main() {
  int n, ans = 0;
  cin >> n;
  vector<string> v(n);

  // Block Logic: Read all operations and update the counter.
  // Pre-condition: 'n' is the number of operations to read.
  for (int i = 0; i < n; i++) {
    cin >> v[i];
    // Check for increment operations. The character '+' appears in both.
    // A more direct check could be `v[i][1] == '+'`.
    if ((v[i] == "X++") || (v[i] == "++X")) {
      ans++;
    } else { // All other operations are decrements.
      ans--;
    }
  }
  
  // Output the final value after all operations.
  cout << ans << endl;
}
