/**
 * @file 281A.CPP
 * @brief "Word Capitalization" - Capitalizes the first letter of a string.
 *
 * This program reads a single word and capitalizes its first letter. If the
 * first letter is already uppercase or is not an alphabet character, it
 * remains unchanged (as per the behavior of `::toupper`).
 *
 * @algorithm
 * The implementation uses standard library functions to achieve the capitalization.
 *
 * 1. Read the entire word into a string `s`.
 * 2. Isolate the first character of `s` by creating a new single-character
 *    string `s1`.
 * 3. Use `std::transform` to apply the `::toupper` function to the new string `s1`.
 *    This converts the first character to its uppercase equivalent in a locale-safe way.
 * 4. Remove the original first character from the input string `s` using `s.erase()`.
 * 5. Concatenate the capitalized first character (`s1`) with the rest of the original
 *    string (`s`) and print the result.
 *
 * @complexity
 * Time Complexity: O(L), where L is the length of the string. The erase operation
 * can take linear time.
 * Space Complexity: O(L) to store the input string.
 */
#include <bits/stdc++.h>
using namespace std;

int main() {
  // Fast I/O setup
  ios::sync_with_stdio(false);
  cin.tie(0);

  // s: the input string, s1: a temporary string for the first character.
  string s, s1;
  cin >> s;

  if (s.empty()) {
    // Handle empty string case to avoid errors.
    return 0;
  }

  // Extract the first character of `s` into `s1`.
  s1 = s.at(0);
  
  // Use std::transform with ::toupper to capitalize the character in `s1`.
  transform(s1.begin(), s1.end(), s1.begin(), ::toupper);
  
  // Remove the original first character from `s`.
  s.erase(0, 1);

  // Concatenate the new capitalized first character with the rest of the string and print.
  cout << s1 + s << "\n";
  
  return 0;
}
