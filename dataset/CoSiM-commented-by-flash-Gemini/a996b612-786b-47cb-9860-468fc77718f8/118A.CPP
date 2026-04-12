/**
 * @file 118A.CPP
 * @brief Logic for string transformation involving vocalic filtering and consonant formatting.
 * 
 * Logic: Processes an input string to remove all vowels (a, e, i, o, u, y), 
 * normalize remaining characters to lowercase, and prefix each with a '.' separator.
 * 
 * Algorithm: Single-pass scan with linear search-based character classification.
 * Time Complexity: $O(N)$ where N is the length of the string (classification is $O(1)$).
 * Space Complexity: $O(N)$ for string storage.
 */

#include <iostream>

using namespace std;

// Static lookup table for vocalic characters to be filtered.
char vowels[] = {'a', 'e', 'i', 'o', 'u', 'y'};

/**
 * @brief Functional Utility: Character classification based on vocalic set membership.
 * @return True if 'c' is in the defined vowel set, False otherwise.
 */
bool isVowel(char c) {
  for (int i = 0; i < 6; i++) {
    if (c == vowels[i]) {
      return true;
    }
  }
  return false;
}

int main() {
  string s;
  // Reads the source character sequence.
  cin >> s;

  /**
   * Block Logic: Main transformation loop.
   * Invariant: Each character is normalized and evaluated for preservation in the output stream.
   */
  for (int i = 0; i < s.length(); i++) {
    // Normalizes uppercase ASCII characters to lowercase.
    s[i] = (char)tolower(s[i]);
    
    /**
     * Logic: Consonant preservation and formatting.
     * If the character is not a vowel, it is prefixed with a dot delimiter.
     */
    if (!isVowel(s[i])) {
      cout << "." << s[i];
    }
  }
  
  // Output termination.
  cout << "\n";
  
  return 0;
}
