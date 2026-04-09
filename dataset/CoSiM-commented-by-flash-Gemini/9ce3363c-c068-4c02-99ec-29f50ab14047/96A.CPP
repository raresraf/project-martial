/**
 * @file 96A.CPP
 * @brief This program determines if a given string is "lucky". A string is considered
 * lucky if it contains at least 7 consecutive identical characters. This is a common
 * string analysis problem in competitive programming.
 *
 * Algorithm: Iterative scan for longest run of identical characters.
 * 1. Read the input string.
 * 2. Iterate through the string, maintaining a count of consecutive identical characters.
 * 3. Keep track of the maximum consecutive count found.
 * 4. If the maximum consecutive count is 7 or more, the string is "lucky".
 * Time Complexity: O(L) where L is the length of the string, due to a single pass.
 * Space Complexity: O(L) for storing the string (or O(1) if input is streamed).
 */

#include <cmath> // Not directly used for this problem, but often included in competitive programming templates.
#include <iostream>
#include <string>

using namespace std;

/**
 * @brief Calculates the maximum length of consecutive identical characters in a string.
 * @param s The input string.
 * @pre `s` is a valid string.
 * @post Returns the maximum count of consecutive identical characters.
 * @return An integer representing the maximum consecutive character count.
 */
int f(string &s) {
  // Functional Utility: `mx` stores the overall maximum consecutive count, `cnt` stores the current consecutive count.
  int mx = 1, cnt = 1;
  // Block Logic: Iterates through the string to find the longest run of identical characters.
  // Invariant: `cnt` accurately tracks the current run length, `mx` holds the maximum found so far.
  // Note: Loop iterates up to `s.length()`, comparing `s[i]` with `s[i-1]`.
  for (int i = 1; i <= s.length(); i++) {
    // Block Logic: Checks if the current character is the same as the previous one.
    if (s[i] == s[i - 1])
      cnt++; // Functional Utility: Increments current consecutive count.
    else
      cnt = 1; // Functional Utility: Resets count if characters differ.
    // Block Logic: Updates `mx` if the current count is greater.
    if (cnt > mx)
      mx = cnt;
  }
  return mx;
}

/**
 * @brief Solves the "lucky string" problem.
 * Reads a string, checks if it contains 7 or more consecutive identical characters,
 * and prints "YES" or "NO".
 * @pre Input consists of a single string.
 * @post Prints "YES" if the string is lucky, "NO" otherwise.
 */
void solve() {
  string s;
  cin >> s; // Functional Utility: Reads the input string.
  // Block Logic: Calls `f` to get the maximum consecutive character count and checks if it's lucky.
  // Pre-condition: `f(s)` returns the maximum consecutive count.
  if (f(s) >= 7)
    cout << "YES
"; // Functional Utility: Prints "YES" if the string is lucky.
  else
    cout << "NO
"; // Functional Utility: Prints "NO" if the string is not lucky.
}

/**
 * @brief Main function of the program.
 * Calls the `solve` function to execute the problem logic.
 * @return 0 upon successful execution.
 */
int main() {
  solve();
  return 0;
}
