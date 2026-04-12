/**
 * @file 118A.CPP
 * @brief Logic for string transformation via vocalic filtering and consonant normalization.
 * 
 * Logic: Implements a specific string processing algorithm:
 * 1. Normalizes characters to lowercase.
 * 2. Filters out all vocalic characters {a, e, i, o, u, y}.
 * 3. Prefixes each remaining consonant with a dot (.) separator.
 * 
 * Algorithm: Single-pass scan with look-ahead append and conditional backtrack.
 * Time Complexity: $O(N)$ where N is the length of the string.
 * Space Complexity: $O(N)$ for the result string storage.
 */

#include <iostream>
#include <string>

using namespace std;

/**
 * @brief Functional Utility: Character classification based on vocalic membership.
 * @return True if 'a' is a defined vowel, False otherwise.
 */
bool vowel(char a) {
  if (a == 'a' || a == 'e' || a == 'i' || a == 'o' || a == 'u' || a == 'y')
    return true;
  return false;
}

int main() {
  string read;
  // Reads the source character sequence.
  cin >> read;
  
  string ans = "";

  /**
   * Block Logic: Iterative transformation of the input string.
   * Logic: Proactively appends a separator and then conditionally removes it 
   * if the subsequent character is classified as a vowel.
   */
  for (int i = 0; i < read.size(); ++i) {
    // Proactive: Appends the consonant delimiter.
    ans += ".";
    
    // Normalization: Converts uppercase ASCII to lowercase.
    char a = (char)tolower(read[i]);
    
    if (vowel(a)) {
      /**
       * Functional Utility: Backtracks if the character is a vowel.
       * Logic: Removes the previously appended '.' as the character is filtered out.
       */
      ans.pop_back();
    } else {
      // Logic: Preserves and appends the normalized consonant.
      ans += a;
    }
  }
  
  // Output the final transformed string.
  cout << ans << endl;
  
  return 0;
}
