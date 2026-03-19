/**
 * @file 58A.CPP
 * @brief Solution to Codeforces Problem 58A: "Chat room".
 *
 * This problem asks to determine if the string "hello" is a subsequence of a given
 * input string `s`. A subsequence is a sequence that can be derived from another
 * sequence by deleting some or no elements without changing the order of the
 * remaining elements.
 *
 * Algorithm: Iterate through the input string `s`. Maintain a pointer/counter
 * for the current character expected from "hello". If a character in `s` matches
 * the expected character from "hello", advance the counter for "hello". If the
 * counter reaches 5 (length of "hello"), then "hello" is a subsequence.
 *
 * Time Complexity: O(L_s), where L_s is the length of the input string `s`.
 *                  This involves a single pass through the string.
 * Space Complexity: O(L_s) to store the input string `s`, plus O(1) for other variables.
 */
#include "bits/stdc++.h" // Includes various standard libraries for C++, like iostream and string.
// PRAY :ORZ FENCING: orz orz orz orz orz          // Original author's personal comments/notes.
// PRAY :DOLPHINGARLIC: orz orz orz orz orz
// PRAY :DORI: orz orz orz orz orz orz orz
// PRAY :SAHIL KUCHLOUS: orz orz orz orz orz
// pray :stefan: orz orz orz orz orz orz orz
// pray :foshy: orz orz orz orz orz orz orz

/** @def all(x)
 * @brief Macro to get iterators to the beginning and end of a container `x`,
 *        commonly used with algorithms like `sort`, `for_each`, etc.
 */
#define all(x) (x).begin(), (x).end()
/** @def rall(x)
 * @brief Macro to get reverse iterators to the reverse beginning and reverse end of a container `x`.
 */
#define rall(x) (x).rbegin(), (x).rend()
/** @def elif
 * @brief Alias for `else if`.
 */
#define elif else if

using namespace std;

/**
 * @brief Main function to solve the "Chat room" problem.
 * Reads an input string, checks if "hello" is its subsequence, and prints "YES" or "NO".
 */
int main() {
  // Functional Utility: Optimizes C++ standard streams for faster input/output.
  ios_base::sync_with_stdio(false);
  cin.tie(NULL); // Unties cin from cout.
  cout.tie(0);   // Prevents flushing cout before cin.

  int cnt{}; // Counter to track matched characters of "hello" (initialized to 0).
  const string h = "hello"; // The target subsequence.
  string s; // The input string to check.
  cin >> s; // Reads the input string.

  /**
   * Block Logic: Iterates through each character of the input string `s` to find "hello" as a subsequence.
   * Pre-condition: `s` contains the input. `cnt` is 0. `h` is "hello".
   * Invariant: `cnt` stores the count of characters of `h` found as a subsequence in `s` so far.
   */
  for (auto i : s) { // Range-based for loop for characters in `s`.
    /**
     * Block Logic: If the current character `i` from `s` matches the next expected character `h[cnt]`
     * from "hello", increment the counter `cnt`.
     */
    if (i == h[cnt]) {
      cnt++; // Increment counter for matched characters.
    }
  }
  // Check if all 5 characters of "hello" were found in order.
  cout << (cnt == 5 ? "YES\n" : "NO\n");
  return 0; // Indicates successful execution.
}
// TASKKILL /F /IM main.exe // Original author's personal comments/notes.
/*
8:56 PM
9/30/2020 m/dd/yyyy
*/
