/**
 * @file 112A.CPP
 * @brief This program performs a case-insensitive lexicographical comparison of two input strings.
 * It converts both strings to lowercase and then compares them character by character
 * to determine their relative order.
 * Algorithm: String case conversion using `std::transform` followed by manual character-by-character comparison.
 * Time Complexity: O(L) where L is the maximum length of the input strings, due to character iteration.
 * Space Complexity: O(L) for storing the input strings.
 */
#include <bits/stdc++.h> // Commonly used header in competitive programming, includes <algorithm> and <iostream>.
#include <iostream>      // Explicitly included for input/output operations (cin, cout).

using namespace std;

int main() {
  string s1, s2; // Stores the two input strings.
  // Reads the two strings from standard input.
  cin >> s1 >> s2;
  // Converts all characters in 's1' to lowercase using std::transform and ::tolower.
  transform(s1.begin(), s1.end(), s1.begin(), ::tolower);
  // Converts all characters in 's2' to lowercase using std::transform and ::tolower.
  transform(s2.begin(), s2.end(), s2.begin(), ::tolower);
  int output = 0; // Initializes 'output' to 0, assuming strings are equal until proven otherwise.
  /**
   * Block Logic: Compares the lowercase strings character by character to determine their lexicographical order.
   * The loop terminates as soon as a difference is found, or when the end of the shorter string is reached.
   * Invariant: 'output' accurately reflects the lexicographical comparison result up to the current index.
   */
  for (int i = 0; i < s1.size(); i++) {
    // If characters at the current position are equal, continue to the next character.
    if (int(s1[i]) == int(s2[i]))
      continue;
    // If the character in 's1' is smaller than in 's2', 's1' is lexicographically smaller.
    if (int(s1[i]) < int(s2[i])) {
      output = -1; // Set output to -1.
      break;       // Terminate comparison as order is determined.
    }
    // If the character in 's1' is larger than in 's2', 's1' is lexicographically larger.
    if (int(s1[i]) > int(s2[i])) {
      output = 1; // Set output to 1.
      break;      // Terminate comparison as order is determined.
    }
  }
  // Prints the comparison result (-1, 0, or 1) to standard output.
  cout << output << endl;

  // Standard program exit.
  return 0;
}
