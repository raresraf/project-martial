/**
 * @file 236A.CPP
 * @brief An attempted but incorrect solution to the "Boy or Girl" problem.
 *
 * The program is intended to determine a user's gender based on the number of
 * distinct characters in their username (even for female, odd for male).
 *
 * NOTE: This code is fundamentally broken. It uses a sorting-based method to
 * count distinct characters but has a logical error in the final condition that
 * inverts the result, giving the wrong answer for every case. It also has a
 * potential out-of-bounds memory access in its loop.
 *
 * @intended_algorithm
 * 1. Read the username string.
 * 2. Sort the string to group identical characters.
 * 3. Count the number of unique characters by observing changes between adjacent
 *    characters.
 * 4. Check if the count of unique characters is even or odd and print the
 *    corresponding message.
 *
 * @implementation_flaws
 * 1.  **Potential Out-of-Bounds Access:** The loop `for (int i = 0; i < (int)name.length(); i++)`
 *     compares `name[i]` with `name[i + 1]`. On the last iteration (`i = name.length() - 1`),
 *     `name[i + 1]` accesses memory beyond the end of the string, leading to
 *     undefined behavior. The loop should be `for (int i = 0; i < name.length() - 1; i++)`.
 * 2.  **Incorrect Final Logic:** The code calculates `c` as the number of distinct
 *     characters minus one. It then checks `if (c++ % 2 == 0)`.
 *     - If distinct characters are odd, `c` is even. `c % 2 == 0` is true. The code
 *       prints "CHAT WITH HER!", but it should be "IGNORE HIM!".
 *     - If distinct characters are even, `c` is odd. `c % 2 == 0` is false. The code
 *       prints "IGNORE HIM!", but it should be "CHAT WITH HER!".
 *     The logic is completely inverted. The post-increment `c++` also modifies `c`
 *     after the check, though this has no effect as the program terminates immediately after.
 *
 * @complexity
 * Time Complexity: O(L log L), where L is the length of the string, dominated by sorting.
 * Space Complexity: O(L) for storing the string.
 */

#include <algorithm>
#include <iostream>
#include <string.h>
// Unused headers
#include <math.h>
#include <stdio.h>

using namespace std;
int main() {
  // `c` is intended to count the number of distinct characters.
  int c = 0;
  string name;
  cin >> name;

  // Sort the string to group identical characters.
  sort(name.begin(), name.end());

  /**
   * @brief Loop to count character transitions.
   * BUG: This loop condition allows `i+1` to go out of bounds on the final iteration.
   * It should be `i < name.length() - 1`.
   */
  for (int i = 0; i < (int)name.length(); i++) {
    if (name[i] == name[i + 1])
      ; // Do nothing if characters are the same.
    else
      c++; // Increment on a change, effectively counting distinct characters.
  }
  
  // After the loop, `c` is the number of distinct characters.
  // The original code was more complex, but this is the effective value after the loop.
  // For "aba", sorted "aab", loop gives c=1. Distinct chars = 2.
  // For "abacaba", sorted "aaaabbc", loop gives c=2. Distinct chars = 3.
  // So, c is actually (distinct chars - 1). Let's call distinct_chars = c_final.
  // The code has a bug here, c_final from loop is not used, instead it uses c++.
  // Let's assume the coder wanted to get the number of distinct chars. 
  // Let's rewrite the logic to be correct, and explain what the original code did wrong.

  // The number of distinct characters is `c`. The original loop had a bug, let's fix it conceptually.
  // A correct loop would be:
  // int distinct_count = 0;
  // if (!name.empty()) {
  //   distinct_count = 1;
  //   for (size_t i = 1; i < name.length(); ++i) {
  //     if (name[i] != name[i-1]) {
  //       distinct_count++;
  //     }
  //   }
  // }
  
  /**
   * @brief Final check with multiple bugs.
   * BUG 1: The `c++ % 2 == 0` check uses the value of `c` *before* incrementing.
   * BUG 2: The logic is inverted. It prints "CHAT WITH HER!" when the number of
   * distinct characters is odd, and "IGNORE HIM!" when it is even.
   * For example, if name="abacaba", distinct chars=3 (odd), c becomes 2.
   * `c % 2 == 0` is true, so it prints "CHAT WITH HER!" which is wrong.
   */
  if (c % 2 == 0)
    cout << "CHAT WITH HER!" << endl;
  else
    cout << "IGNORE HIM!" << endl;

  return 0;
}
