/**
 * @file 58A.CPP
 * @brief Determines if the word "hello" can be formed by deleting characters from an input string.
 *
 * This program checks if the string "hello" is a subsequence of a given string `line` read from
 * standard input. It iterates through the input string, trying to match the characters of "hello"
 * in order.
 *
 * @algorithm Subsequence Matching
 * 1. Initialize a target string `greeting` as "hello".
 * 2. Read the source string `line` from the user.
 * 3. Use two pointers: `a` for the index in `greeting` and `b` for the index in `line`.
 * 4. Iterate through `line` with pointer `b`.
 * 5. If the character `line[b]` matches `greeting[a]`, increment `a` to look for the next character of "hello".
 * 6. Always increment `b` to move to the next character in `line`.
 * 7. If `a` reaches the length of `greeting`, it means all characters of "hello" have been found in order.
 *    The program can then terminate early and report "YES".
 * 8. If the loop finishes and `a` has not reached the length of `greeting`, "hello" is not a subsequence,
 *    and the program reports "NO".
 *
 * @complexity
 * Time Complexity: O(L), where L is the length of the input string `line`, because we iterate through it once.
 * Space Complexity: O(L) to store the input string. The space for the target string is constant.
 */
#include <iostream>
#include <string> // It's good practice to include the <string> header for std::string.
using namespace std;

int main() {
  // The target subsequence to find.
  const string greeting = "hello";
  // The input string from which to form the greeting.
  string line;
  // Read the entire line of input.
  getline(cin, line);

  // 'a' is a pointer to the current character in 'greeting'.
  int a = 0;
  // 'b' is a pointer to the current character in 'line'.
  int b = 0;
  // The final result, initialized to "NO".
  string output = "NO";

  /**
   * @brief Iterates through the input string `line` to find the `greeting` subsequence.
   *
   * Invariant: `a` stores the number of characters from `greeting` that have been matched so far.
   */
  while (b < line.size()) {
    // If the current character in `line` matches the current character we're looking for in `greeting`.
    if (line[b] == greeting[a]) {
      // Move to the next character in `greeting`.
      a++;
    }
    // Always move to the next character in `line`.
    b++;

    // Pre-condition for success: if we have matched all characters in `greeting`.
    if (a >= greeting.size()) {
      // Update the result and exit the loop.
      output = "YES";
      break;
    }
  }

  // Print the final result.
  cout << output << endl;
  return 0;
}
