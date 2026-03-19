/**
 * @file 58A.CPP
 * @brief Solution to Codeforces Problem 58A: "Chat room".
 *
 * This problem asks to determine if the string "hello" is a subsequence of a given
 * input string `str`. A subsequence is a sequence that can be derived from another
 * sequence by deleting some or no elements without changing the order of the
 * remaining elements.
 *
 * Algorithm: Iterate through the input string `str` using a pointer (`i`), and
 * simultaneously maintain a pointer (`j`) for the characters of "hello". If a character
 * in `str` matches the character `hello[j]`, increment `j`. If `j` reaches 5, "hello"
 * is a subsequence.
 *
 * Time Complexity: O(L_s), where L_s is the length of the input string `str`,
 *                  as it involves a single pass through the string.
 * Space Complexity: O(L_s) to store the input string `str`, plus O(1) for other variables.
 *
 * Author: Ikki
 * Date: Jun 10, 2015
 */
#include <iostream> // Required for input/output operations (cin, cout).
#include <string>   // Required for string manipulation.
using namespace std;

/**
 * @brief Main function to solve the "Chat room" problem.
 * Reads an input string, checks if "hello" is its subsequence, and prints "YES" or "NO".
 */
int main() {
  string str; // Stores the input string from the user.
  cin >> str; // Reads the input string.
  
  string hello = "hello"; // The target subsequence string.
  int j = 0; // Pointer/index for the 'hello' string, tracking matched characters.

  /**
   * Block Logic: Iterates through the input string `str` to find "hello" as a subsequence.
   * Pre-condition: `str` contains the input. `hello` is "hello". `j` is 0.
   * Invariant: `j` stores the count of characters of "hello" found as a subsequence in `str` so far.
   * The loop continues as long as `i` is within `str`'s bounds and `j` has not yet matched all of "hello".
   */
  for (int i = 0; i < (int)str.size() && j < 5; i++) {
    /**
     * Block Logic: If the current character `str[i]` matches the next expected character `hello[j]`,
     * increment `j` to look for the subsequent character of "hello".
     */
    if (str[i] == hello[j]) {
      j++; // Advance the 'hello' string pointer/index.
    }
  }
  
  // After the loop, check if all 5 characters of "hello" were found in order.
  // Uses a ternary operator to print "YES" if `j` is 5, otherwise "NO".
  cout << (j == 5 ? "YES" : "NO") << endl;
  return 0; // Indicates successful execution.
}