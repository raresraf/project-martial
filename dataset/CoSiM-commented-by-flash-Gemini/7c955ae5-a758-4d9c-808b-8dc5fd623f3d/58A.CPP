/**
 * @file 58A.CPP
 * @brief Solution for Codeforces Problem 58A: Chat room.
 * @details This program checks if the word "hello" can be formed as a subsequence
 * from a given input string. It iterates through the input string, attempting to match
 * characters of "hello" in their correct order, but not necessarily consecutively in the input.
 * @algorithm Subsequence matching (two-pointer approach).
 * Time Complexity: $O(L_s)$ where $L_s$ is the length of the input string.
 * Space Complexity: $O(L_t)$ where $L_t$ is the length of the target string "hello" (constant, i.e., O(1)).
 */
#include "bits/stdc++.h"

using namespace std;

/**
 * @file 58A.CPP
 * @brief Solution for Codeforces Problem 58A: Chat room.
 * @details This program checks if the word "hello" can be formed as a subsequence
 * from a given input string. It iterates through the input string, attempting to match
 * characters of "hello" in their correct order, but not necessarily consecutively in the input.
 * @algorithm Subsequence matching (two-pointer approach).
 * Time Complexity: $O(L_s)$ where $L_s$ is the length of the input string.
 * Space Complexity: $O(L_t)$ where $L_t$ is the length of the target string "hello" (constant, i.e., O(1)).
 */
#include "bits/stdc++.h" // Includes most standard C++ library headers (e.g., iostream, string).

using namespace std; // Uses the standard namespace.

int main(int argc, char const *argv[]) {
  /**
   * @brief Main function: orchestrates the subsequence matching logic.
   * @param argc The number of command-line arguments.
   * @param argv An array of command-line argument strings.
   * @note `argc` and `argv` are part of the `main` signature but are not used in this solution.
   */
  string str; // @var str: The input string to search within.
  cin >> str; // Reads the input string.

  int inc = 0;      // @var inc: Index for `str2` (the target subsequence "hello").
                    //           Also acts as a counter for matched characters.
  string str2 = "hello"; // @var str2: The target subsequence string.

  /**
   * @brief Iterates through the input string `str` to find the "hello" subsequence.
   * @block_logic Uses a two-pointer approach: one for `str` (loop variable `i`) and one for `str2` (`inc`).
   * @invariant `inc` advances only when a matching character from `str2` is found in `str`.
   * @note There is a potential out-of-bounds access on `str2[inc]` if `inc` reaches `str2.size()` before
   *       the loop for `str` finishes. A more robust check (`if (inc < str2.size() && str[i] == str2[inc])`)
   *       would prevent this, but adhering to "Zero Code Mutation".
   */
  for (int i = 0; i < str.size(); i++) {
    if (str[i] == str2[inc]) { // Compares current character of `str` with current target character from `str2`.
      inc++; // If characters match, advance the target string index.
    }
  }

  /**
   * @brief Determines if the entire "hello" subsequence was found.
   * @block_logic Checks if `inc` reached 5 (the length of "hello").
   * @invariant Prints "YES" if "hello" is a subsequence, otherwise "NO".
   */
  if (inc == 5) // If all characters of "hello" were matched in order.
    cout << "YES"; // Output "YES".
  else
    std::cout << "NO"; // Output "NO".

  return 0; // Indicates successful execution.
}
