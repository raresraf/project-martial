/**
 * @file 58A.CPP
 * @brief Solution for Codeforces Problem 58A, "Chat room".
 * @details The program checks if the word "hello" is a subsequence of a given
 *          input string. This is determined by iterating through the input string
 *          and greedily matching the characters of "hello" in order.
 *
 * @algorithm Greedy Subsequence Search
 * A pointer `j` tracks the current character to be found in the target string "hello".
 * The code iterates through the input string `a`. If the character `a[i]` matches
 * the target character `b[j]`, the pointer `j` is advanced. If `j` successfully
 * traverses all of "hello", the subsequence exists.
 *
 * @complexity
 * Time: O(N), where N is the length of the input string `a`, due to the single pass.
 * Space: O(N) to store the input string `a`.
 */
#include <iostream>
using namespace std;

// --- Global variables for the solution ---
string a;             // The input string from the user.
string b = "hello";   // The target subsequence.
int j = 0;            // Pointer for the current character in the target string `b`.
int pas = 0;          // Counter for matched characters. Redundant, as `j` serves the same purpose.

int main() {
  cin >> a;

  /**
   * @brief This loop performs a greedy, single-pass search for the subsequence "hello".
   * @invariant `j` points to the next character to be found in "hello". `pas` counts
   *            how many characters have been successfully matched in order.
   */
  for (int i = 0; i < a.size(); i++) {
    // If the current character in the input matches the target character...
    if (a[i] == b[j]) {
      // ...advance the target pointer and increment the match counter.
      j++;
      pas++;

      // Optimization: if all 5 characters are found, no need to search further.
      if (pas == 5) {
        break;
      }
    }
  }

  // If the total number of matched characters is 5, then "hello" is a subsequence.
  if (pas == 5) {
    cout << "YES";
  } else {
    cout << "NO";
  }
}
