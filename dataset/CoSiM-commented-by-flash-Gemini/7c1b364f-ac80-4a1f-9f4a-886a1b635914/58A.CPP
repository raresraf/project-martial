/**
 * @file 58A.CPP
 * @brief Solution to a competitive programming problem that checks if the word "hello"
 * can be formed as a subsequence from a given input string.
 * A subsequence is a sequence that can be derived from another sequence by deleting
 * some or no elements without changing the order of the remaining elements.
 * Algorithm: Two-pointer approach (or linear scan with state-tracking) for subsequence matching.
 * Time Complexity: O(L), where L is the length of the input string.
 * Space Complexity: O(1) for storing the target string and indices.
 */
#include <bits/stdc++.h> // Includes most standard C++ libraries.
using namespace std;

int main() {
  string s, target; // Declares 's' for the input string and 'target' for the subsequence to find.
  // Reads the input string from standard input.
  cin >> s;

  target = "hello"; // Sets the target subsequence to "hello".
  int n = s.length(); // Stores the length of the input string.

  int k = 0; // 'k' is an index for the 'target' string, tracking characters matched so far.
  /**
   * Block Logic: Iterates through the input string to find characters of the target subsequence ("hello").
   * Invariant: At the start of each iteration, 'i' is the current character's index in `s`,
   * and 'k' is the number of characters of `target` that have been matched as a subsequence.
   * Pre-condition: 's' contains a valid string.
   */
  for (int i = 0; i < n; i++) {
    // Conditional Logic: If the current character in 's' matches the next expected character in 'target'.
    if (s[i] == target[k]) {
      k++; // Advances the target string index.
    }
  }
  
  /**
   * Conditional Logic: After iterating through the input string, checks if all characters of "hello" were matched.
   * If 'k' is equal to the length of 'target', it means "hello" was found as a subsequence.
   */
  if (k == target.length()) {
    cout << "YES" << endl; // Prints "YES".
  } else {
    cout << "NO" << endl;  // Prints "NO".
  }

  return 0; // Indicates successful program execution.
}
