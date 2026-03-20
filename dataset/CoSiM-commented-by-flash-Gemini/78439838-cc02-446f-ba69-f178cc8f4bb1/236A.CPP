/**
 * @file 236A.CPP
 * @brief Solution to a string uniqueness problem (Codeforces problem 236A - Boy or Girl).
 *        This variant solves the problem by counting distinct characters using an unordered_set.
 * Algorithm:
 *   1. Read the input string (username).
 *   2. Iterate through each character of the string and insert it into an `unordered_set`.
 *      The `unordered_set` automatically handles uniqueness, storing only one instance of each character.
 *   3. Get the size of the `unordered_set`, which represents the count of distinct characters.
 *   4. Check the parity (even or odd) of this count.
 *   5. Print "CHAT WITH HER!" if the count is odd, and "IGNORE HIM!" if the count is even.
 * Time Complexity: O(N) on average, where N is the length of the input string, due to hash table operations
 *                  for `unordered_set` insertion. In the worst case, it can be O(N^2).
 * Space Complexity: O(K), where K is the number of distinct characters (at most 26 for lowercase English letters).
 */
#include <algorithm>
#include <cmath>
#include <iostream>
#include <string>
#include <unordered_set>
#include <vector>

using namespace std;

int main() {
  /**
   * @brief Main function to solve the "Boy or Girl" problem.
   *
   * Reads a username string, counts the number of distinct characters in it
   * using an `unordered_set`, and then prints "CHAT WITH HER!" if the count
   * of distinct characters is odd, or "IGNORE HIM!" if the count is even.
   * @return int Returns 0 upon successful execution.
   */
  string a;
  unordered_set<char> b;
  cin >> a;
  // Block Logic: Populates an unordered_set with characters from the input string.
  // Invariant: 'b' will contain only unique characters from 'a'.
  for (char c : a) {
    b.insert(c);
  }
  // Block Logic: Determines the output message based on the parity of unique characters.
  // Pre-condition: 'b.size()' holds the count of distinct characters.
  // Post-condition: Prints "CHAT WITH HER!" if the count is even, "IGNORE HIM!" if odd.
  if (b.size() % 2 == 0) {
    cout << "CHAT WITH HER!";
  } else {
    cout << "IGNORE HIM!";
  }
}