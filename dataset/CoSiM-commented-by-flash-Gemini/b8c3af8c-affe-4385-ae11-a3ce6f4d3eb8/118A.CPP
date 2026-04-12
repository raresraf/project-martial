/**
 * @file 118A.CPP
 * @brief Logic for string transformation involving vocalic filtering and consonant formatting.
 * 
 * Logic: Implements a specific string processing algorithm:
 * 1. Normalizes characters to lowercase.
 * 2. Filters out all vocalic characters based on an unordered set of {a, i, u, e, o, y}.
 * 3. Prefixes each remaining character (consonant) with a dot (.) separator.
 * 
 * Algorithm: Single-pass scan with hash-set based character classification.
 * Time Complexity: $O(N)$ where N is the length of the string (average set lookup is O(1)).
 * Space Complexity: $O(N)$ for the input string buffer.
 */

#include <iostream>
#include <iterator>
#include <string>
#include <unordered_set>

using std::cin;
using std::cout;
using std::endl;
using std::string;

int main() {
  string input;
  // Functional Utility: Reads the source character sequence.
  if (!(cin >> input)) return 0;

  // Data Structure: Unordered set for $O(1)$ average time classification of vowels.
  std::unordered_set<char> vowels = {'a', 'i', 'u', 'e', 'o', 'y'};

  /**
   * Block Logic: Iterative transformation of the input string.
   * Invariant: Evaluates each character independently for normalization and preservation.
   */
  for (auto it = input.begin(); it != input.end(); ++it) {
    /**
     * Logic: In-place case normalization.
     * Maps uppercase ASCII characters to their lowercase counterparts.
     */
    (*it) = (char)tolower(*it);

    /**
     * Functional Utility: Conditional filtering.
     * Logic: If character is not found in the vocalic set, it is output with 
     * a preceding '.' delimiter.
     */
    if (vowels.find(*it) == vowels.end()) {
      cout << "." << *it;
    }
  }
  
  return 0;
}
