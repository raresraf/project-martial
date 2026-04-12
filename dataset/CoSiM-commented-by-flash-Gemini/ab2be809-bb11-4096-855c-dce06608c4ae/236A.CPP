/**
 * @file 236A.CPP
 * @brief Binary classification based on unique character cardinality in a string.
 * 
 * Logic: Determines the class of a user ('HER' or 'HIM') based on the parity 
 * of the count of distinct characters in their username. An even count results 
 * in 'CHAT WITH HER!', an odd count results in 'IGNORE HIM!'.
 * 
 * Algorithm: Sorting followed by a single-pass adjacency comparison.
 * Time Complexity: $O(N \log N)$ where N is the length of the string.
 * Space Complexity: $O(N)$ for string storage.
 */

#include <algorithm>
#include <iostream>
using namespace std;

int main() {
  string str;
  // Functional Utility: Reads the user identifier string.
  cin >> str;

  // Cardinality of the set of unique characters.
  int counter = 0;

  // Logic: Groups identical characters to simplify distinct count calculation.
  sort(str.begin(), str.end());

  /**
   * Block Logic: Identifies transitions between unique character groups.
   * Invariant: Increments the counter whenever the current character 
   * differs from the next one (including the transition to the null terminator).
   */
  for (unsigned int i = 0; i < str.length(); i++) {
    if (str[i] != str[i + 1]) {
      counter++;
    }
  }

  /**
   * Functional Utility: Maps the parity of the unique set size to a binary classification.
   */
  if (counter % 2 == 0) {
    cout << "CHAT WITH HER!";
  } else {
    cout << "IGNORE HIM!";
  }
  
  return 0;
}
