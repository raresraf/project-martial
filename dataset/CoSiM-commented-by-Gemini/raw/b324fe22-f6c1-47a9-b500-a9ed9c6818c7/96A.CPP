/**
 * @file 96A.CPP
 * @brief Solution to Codeforces problem 96A - "Football".
 * @details The program checks for a "dangerous" situation, defined as at least
 *          seven consecutive identical characters ('0' or '1') in a string.
 *          This version uses a single-pass counter approach.
 *
 * Algorithm:
 * 1. Read the input string.
 * 2. Initialize a counter `c1` for the current run of identical characters.
 * 3. Iterate through the string, comparing each character to the next.
 * 4. If a character is the same as the next one, increment the counter.
 *    - If the counter reaches 7, break the loop as the condition is met.
 * 5. If a character is different from the next, reset the counter to 1.
 * 6. After the loop, if the counter is 7, print "YES"; otherwise, print "NO".
 *
 * Note: The implementation contains unused variables (`flag1`, `flag2`, `c2`).
 * The loop `for (int i = 0; i < a.length(); i++)` also creates a potential
 * out-of-bounds read when `i` is at its last value, as `a[i + 1]` is accessed.
 *
 * Time Complexity: O(N), for a single pass over the string of length N.
 * Space Complexity: O(N), to store the input string.
 */
#include <iostream>
#include <string>
using namespace std;

int main() {
  string a;
  // Unused variables `flag1`, `flag2`, `c2` are declared but not used.
  int flag1 = 0, flag2 = 0, c1 = 1, c2 = 0;
  cin >> a;
  /**
   * @brief Main loop to find a run of 7 identical characters.
   * Note: The condition `i < a.length()` could lead to an out-of-bounds read at `a[i+1]`
   * on the final iteration.
   */
  for (int i = 0; i < a.length(); i++) {
    // If the current character matches the next, extend the current run.
    if (a[i] == a[i + 1]) {
      c1++;
      // If a run of 7 is found, no need to search further.
      if (c1 == 7)
        break;
    } else {
      // Otherwise, the run is broken; reset the counter.
      c1 = 1;
    }
  }
  
  // Post-loop check to see if a dangerous situation was found.
  if (c1 == 7)
    cout << "YES" << endl;
  else
    cout << "NO" << endl;

  return 0;
}
