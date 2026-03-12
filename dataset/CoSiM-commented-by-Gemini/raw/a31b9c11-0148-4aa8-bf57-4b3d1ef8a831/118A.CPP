/**
 * @file 118A.CPP
 * @brief Solution to the "String Task" problem.
 *
 * This program processes a string based on a set of rules:
 * 1. It converts all uppercase letters to lowercase.
 * 2. It removes all vowels ('a', 'o', 'y', 'e', 'u', 'i').
 * 3. It inserts a '.' character before every consonant.
 *
 * @algorithm Linear Scan
 * @time_complexity O(N), where N is the length of the input string. This is due to the `transform` function and the loop, both of which are linear.
 * @space_complexity O(N) to store the input string.
 */
#include <bits/stdc++.h>
using namespace std;

int main() {
  // Optimizes I/O operations for faster execution.
  ios::sync_with_stdio(false);
  cin.tie(NULL);
  cout.tie(NULL);

  string s;
  cin >> s;

  // Converts the entire string to lowercase to handle case-insensitivity.
  transform(s.begin(), s.end(), s.begin(), ::tolower);

  /**
   * @brief Iterates through the modified string to apply the task rules.
   *
   * Pre-condition: The string `s` is entirely in lowercase.
   * Logic: For each character, if it is a vowel, it is skipped.
   * If it is a consonant, a '.' is printed, followed by the consonant itself.
   */
  for (char c : s)
    if (c == 'a' || c == 'e' || c == 'i' || c == 'o' || c == 'u' || c == 'y')
      continue;
    else
      cout << "." << c;
  cout << "\n";
}