/**
 * @file 58A.CPP
 * @brief Solves the "Chat room" problem from Codeforces (58A).
 * @details The program determines if the word "hello" is a subsequence of a given
 * input string. This means it checks if "hello" can be obtained by deleting
 * zero or more characters from the input string.
 *
 * @section Algorithm
 * A greedy two-pointer approach is used to solve this subsequence problem efficiently.
 * 1. A target string `greeting` is defined as "hello".
 * 2. A pointer `a` tracks the current character being sought in the `greeting` string.
 * 3. A second pointer `b` iterates through the input `line`.
 * 4. The `line` is scanned from left to right. When the character at `line[b]`
 *    matches the character at `greeting[a]`, the `a` pointer is advanced to
 *    look for the next character of the `greeting`.
 * 5. If the `a` pointer successfully traverses all characters of "hello", it means
 *    the subsequence was found in the correct order. The result is "YES".
 * 6. If the end of the input `line` is reached before "hello" is fully spelled,
 *    the result is "NO".
 *
 * @section Complexity
 * - Time Complexity: O(L), where L is the length of the input string, as we
 *   perform a single pass through it.
 * - Space Complexity: O(L) to store the input string.
 */
#include <iostream>
using namespace std;

int main() {
  /**
   * @var greeting The target subsequence string.
   * @var line The input string provided by the user.
   * @var a An index for the `greeting` string, acting as a pointer to the current target character.
   * @var b An index for the `line` string, used to iterate through the input.
   * @var output Stores the final result, initialized to "NO".
   */
  const string greeting = "hello";
  string line;
  getline(cin, line);
  int a = 0, b = 0;
  string output = "NO";

  /**
   * @brief Block Logic: This loop implements the greedy two-pointer search.
   * It scans the input `line` (with pointer `b`) to find the characters of
   * `greeting` (tracked by pointer `a`) in order.
   */
  while (b < line.size()) {
    /**
     * @brief Logic: If the current character in the input matches the target
     * character in "hello", advance the target pointer `a`.
     */
    if (line[b++] == greeting[a]) {
      a++;
    }
    /**
     * @brief Logic: Check if all characters of "hello" have been found.
     * If pointer `a` has passed the last index, the subsequence is complete.
     */
    if (a >= greeting.size()) {
      output = "YES";
      break; // Exit early as the condition has been met.
    }
  }

  // Print the final result.
  cout << output << endl;
  return 0;
}