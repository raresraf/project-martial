/**
 * @file 236A.CPP
 * @brief Solution to Codeforces problem 236A - Boy or Girl.
 *        This program determines if a given username belongs to a "boy" or a "girl"
 *        based on the parity (even or odd) of the number of distinct characters in the username.
 *        If the number of distinct characters is even, it's a "girl" ("CHAT WITH HER!").
 *        If the number of distinct characters is odd, it's a "boy" ("IGNORE HIM!").
 *
 * Algorithm: Counting Distinct Characters
 *   1. Reads the input username string `s`.
 *   2. Uses a hash map (std::map<char, int>) to store the frequency of each character.
 *   3. Iterates through the username string:
 *      a. For each character, increments its count in the map.
 *      b. If the character's count becomes 1 (meaning it's the first time this distinct
 *         character is encountered), increments a `distinct_count` variable.
 *   4. After processing the entire string, `distinct_count` holds the total number
 *      of unique characters.
 *   5. Checks the parity of `distinct_count`:
 *      a. If `distinct_count` is even, prints "CHAT WITH HER!".
 *      b. If `distinct_count` is odd, prints "IGNORE HIM!".
 *
 * Time Complexity: O(L), where L is the length of the input string `s`.
 *                  Map operations (insertion, lookup) take average O(1) time for character keys.
 * Space Complexity: O(C), where C is the size of the character set (e.g., 26 for English alphabet)
 *                   for the `std::map`. This is effectively O(1) as C is a constant.
 *                   O(L) for storing the input string `s`.
 */

#include <bits/stdc++.h> // Includes common C++ standard library headers.
#include <map>           // Required for std::map.
using namespace std;

/**
 * @brief Implements the core logic to solve the "Boy or Girl" problem.
 *
 * Reads a username, counts its distinct characters, and prints the appropriate message.
 */
void solve() {
  map<char, int> m; // Hash map to store character frequencies.
  string s;         // Input username string.
  cin >> s;
  int n = s.length(); // Length of the username string.
  int dem = 0;        // Counter for distinct characters.

  /**
   * Block Logic: Iterates through the username string to count distinct characters.
   * Invariant: `m` stores frequencies, `dem` stores the count of distinct characters
   *            encountered up to the current position.
   */
  for (int i = 0; i < n; i++) {
    m[s[i]]++; // Increments the frequency of the current character.
    /**
     * Conditional Logic: If this is the first occurrence of `s[i]`, increment `dem`.
     * Pre-condition: `m[s[i]]` has just been incremented.
     */
    if (m[s[i]] == 1) {
      dem++; // `dem` counts the number of unique characters.
    }
  }
  
  /**
   * Conditional Logic: Checks the parity of `dem` (number of distinct characters).
   * Prints "CHAT WITH HER!" if even, "IGNORE HIM!" if odd.
   */
  if (dem % 2 == 0) { // If the number of distinct characters is even.
    cout << "CHAT WITH HER!";
  } else { // If the number of distinct characters is odd.
    cout << "IGNORE HIM!";
  }
}

int main() {
  /**
   * @brief Main function.
   *
   * Calls the `solve` function to execute the problem logic.
   *
   * @return int Returns 0 upon successful execution.
   */
  solve();
  return 0;
}
