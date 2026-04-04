/**
 * @file 58A.CPP
 * @brief Determines if the word "hello" can be formed as a subsequence from a given input string.
 * * Algorithm: Greedy Subsequence Matching.
 * Time Complexity: $O(N)$ where N is the length of the input string, due to a single pass.
 * Space Complexity: $O(1)$ for storing auxiliary variables and the target string "hello".
 *
 * This program checks if the characters 'h', 'e', 'l', 'l', 'o' appear in the correct
 * order within the input string, not necessarily contiguously. It uses a greedy approach
 * to find the shortest subsequence match.
 */

#include <iostream> // Standard input/output stream library.
using namespace std;

int main() {
  // Declaration of variables.
  // i: Loop counter for iterating through the input string.
  // j: Initialized to 10, but unused in the provided logic.
  // k: Counter for the number of characters of "hello" successfully matched so far.
  // n: Index for iterating through the target string "hello" (b[]).
  // a[101]: Character array to store the input string.
  // b[]: Character array storing the target subsequence "hello".
  int i, j = 10, k = 0, n = 0;
  char a[101], b[] = "hello";
  cin >> a; // Reads the input string into character array `a`.

  /**
   * Block Logic: Iterates through the input string `a` to greedily find the characters
   * of "hello" in sequential order.
   * Pre-condition: `a` contains the input string. `k` and `n` are initialized to 0.
   * Invariant: `k` always counts the number of matched characters of "hello" from its beginning.
   *            `n` points to the next character in "hello" to be matched.
   */
  for (i = 0; a[i] != '\0'; i++) {
    // Block Logic: If the current character in `a` matches the next expected character in "hello".
    if (a[i] == b[n]) {
      k++;   // Increment the count of matched characters.
      n++;   // Move to the next character in "hello" to search for.
    }
    // Block Logic: If all 5 characters of "hello" have been matched.
    if (k == 5) {
      cout << "YES"; // Print "YES" and terminate the program.
      return 0;
    }
  }
  // Block Logic: If the loop finishes and "hello" was not fully matched (k is not 5).
  if (k != 5)
    cout << "NO"; // Print "NO".
  return 0; // Standard successful exit code for main function.
}
