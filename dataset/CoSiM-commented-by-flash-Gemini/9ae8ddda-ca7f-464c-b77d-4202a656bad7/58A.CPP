/**
 * @file 58A.CPP
 * @brief This program checks if the word "hello" can be formed as a subsequence
 * of a given input string. This means all characters of "hello" must appear
 * in the input string in their correct order, but not necessarily consecutively.
 * This is a classic string subsequence matching problem in competitive programming.
 *
 * Algorithm: Two-pointer subsequence matching.
 * 1. Define the target subsequence "hello".
 * 2. Read the input string.
 * 3. Use two pointers: `strindex` for the input string and `checker` for "hello".
 * 4. Iterate through the input string: if the character at `strindex` matches the
 *    character at `checker`, increment `checker`.
 * 5. After iterating through the input string, if `checker` has reached the length
 *    of "hello" (i.e., 5), then "hello" is a subsequence.
 * 6. Print "YES" or "NO" accordingly.
 * Time Complexity: O(L_input) where L_input is the length of the input string, as it involves a single pass.
 * Space Complexity: O(L_target) for storing the target string "hello".
 */

// CodeForces
// Problem 58A [Chat Room]

#include <bits/stdc++.h>
using namespace std;

/**
 * @brief Main function of the program.
 * Reads an input string and determines if "hello" is a subsequence within it.
 * @pre Input consists of a single string.
 * @post Prints "YES" if "hello" is a subsequence, "NO" otherwise.
 * @return 0 upon successful execution.
 */
int main() {
  // Functional Utility: Stores the target subsequence string "hello".
  string hello = "hello";

  // Functional Utility: Stores the input string provided by the user.
  string inputstr;
  cin >> inputstr;

  // Functional Utility: `strindex` is the pointer for iterating through `inputstr`.
  // Functional Utility: `checker` is the pointer for tracking matched characters in `hello`.
  unsigned int strindex, checker = 0;
  // Block Logic: Iterates through the input string to find characters of "hello" in sequence.
  // Invariant: `checker` counts the number of characters of `hello` found so far in `inputstr` in correct order.
  for (strindex = 0; strindex < inputstr.length(); strindex++) {
    // Block Logic: If the current character in `inputstr` matches the next expected character in `hello`.
    if (inputstr[strindex] == hello[checker]) {
      checker++; // Functional Utility: Advances the `hello` pointer, indicating a match.
    }
  }
  // Block Logic: Determines and prints the final result based on whether all characters of "hello" were matched.
  // Pre-condition: `checker` holds the total count of matched characters from "hello".
  if (checker >= 5) { // Functional Utility: 5 is the length of "hello". If `checker` is 5, all characters matched.
    cout << "YES";
  } else {
    cout << "NO";
  }
  return 0;
}
