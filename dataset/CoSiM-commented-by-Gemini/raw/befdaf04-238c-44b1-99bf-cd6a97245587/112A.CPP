/**
 * @file 112A.CPP
 * @brief Solution to a competitive programming problem, likely Codeforces 112A "Petya and Strings".
 *
 * @details
 * This program performs a case-insensitive lexicographical comparison of two input
 * strings. It prints -1 if the first string is lexicographically smaller, 1 if
 * it is larger, and 0 if they are equal.
 *
 * @algorithm
 * 1. Read two strings from standard input.
 * 2. Normalize both strings to lowercase using the `std::transform` algorithm. This is an
 *    idiomatic C++ way to apply an operation to a range.
 * 3. Manually iterate through the strings character by character, comparing them
 *    at each position.
 * 4. At the first point of difference, determine the lexicographical order, set the
 *    output value (-1 or 1), and break the loop.
 * 5. If the loop completes without finding any differences, the strings are equal,
 *    and the output remains 0.
 *
 * @complexity
 * - Time Complexity: O(N), where N is the length of the strings. This is composed of
 *   O(N) for the two `transform` calls and O(N) for the comparison loop.
 * - Space Complexity: O(N) to store the two input strings.
 */

#include <bits/stdc++.h>
#include <iostream>

using namespace std;

/**
 * @brief Main function and entry point of the program.
 */
int main() {
  string s1, s2;
  cin >> s1 >> s2;

  /**
   * @brief Case normalization using std::transform.
   * This applies the `::tolower` function to every character in each string,
   * converting them in-place.
   */
  transform(s1.begin(), s1.end(), s1.begin(), ::tolower);
  transform(s2.begin(), s2.end(), s2.begin(), ::tolower);
  
  // `output` is initialized to 0, the default value for equal strings.
  int output = 0;

  /**
   * @brief Character-by-character comparison.
   * The loop searches for the first position where the strings differ.
   */
  for (int i = 0; i < s1.size(); i++) {
    // If characters are the same, continue to the next pair.
    if (int(s1[i]) == int(s2[i]))
      continue;
    
    // If a character in s1 is smaller, s1 is lexicographically smaller.
    if (int(s1[i]) < int(s2[i])) {
      output = -1;
      break; // Exit loop as soon as the order is determined.
    }
    
    // If a character in s1 is larger, s1 is lexicographically larger.
    if (int(s1[i]) > int(s2[i])) {
      output = 1;
      break; // Exit loop.
    }
  }
  
  cout << output << endl;

  return 0;
}
