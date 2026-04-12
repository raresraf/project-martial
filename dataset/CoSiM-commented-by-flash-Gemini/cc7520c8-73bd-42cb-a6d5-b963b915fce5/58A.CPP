/**
 * @file 58A.CPP
 * @brief Solution to the "Chat room" problem.
 *
 * This program checks if the word "hello" can be formed as a subsequence from a given input string.
 * This implies that the characters 'h', 'e', 'l', 'l', 'o' must appear in the input string
 * in that specific order, not necessarily consecutively.
 *
 * Algorithm: Subsequence Matching.
 * Time Complexity: O(L) where L is the length of the input string, as it performs a single pass.
 * Space Complexity: O(1) beyond the input string and the small 'hello' array.
 */
#include <iostream>
#include <string>
#define LOG(x) cout << x << "\n"

using namespace std;

bool parseHello(string);

/**
 * @brief Main function to initiate the "hello" subsequence check.
 *
 * Functional Utility: Reads the input string, calls `parseHello` to check for the subsequence,
 * and prints "YES" or "NO" based on the result.
 */
int main() {
  string str;        // Declare a string to store the input.
  cin >> str;        // Read the input string.
  // Call parseHello and print "YES" if it returns true, "NO" otherwise, using the LOG macro.
  parseHello(str) ? LOG("YES") : LOG("NO");
  return 0;
}

/**
 * @brief Checks if "hello" is a subsequence of the given string.
 *
 * This function iterates through the input string `s`, attempting to find the
 * characters of "hello" in sequential order.
 *
 * @param s The input string to search within.
 * @return `true` if "hello" is found as a subsequence, `false` otherwise.
 */
bool parseHello(string s) {
  char hello[] = {'h', 'e', 'l', 'l', 'o'}; // The target subsequence "hello".
  int i = 0; // Index to track the current character being sought in 'hello'.

  /**
   * Block Logic: Iterates through each character `c` of the input string `s`.
   * It tries to match `c` with `hello[i]`. If a match occurs, `i` is incremented,
   * moving to the next character in "hello".
   * Pre-condition: `s` is a valid string.
   * Invariant: `i` indicates how many characters of "hello" have been successfully matched so far.
   */
  for (char c : s) {
    // If we haven't found all characters of "hello" yet (i < 5) and the current character 'c' matches
    // the next character we are looking for in 'hello'.
    if (i < 5 && c == hello[i]) {
      i++; // Move to the next character in "hello" to search for.
    }
    // If all 5 characters of "hello" have been found in the correct order.
    if (i == 5) {
      return true; // "hello" is a subsequence.
    }
  }
  return false; // "hello" was not found as a subsequence.
}