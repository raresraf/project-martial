/**
 * @file 118A.CPP
 * @brief Solution for competitive programming problem: "String Task".
 *
 * @details This program transforms an input string by first removing all vowels
 *          (A, O, Y, E, U, I, and their lowercase counterparts). Then, for each
 *          remaining consonant, it prepends a '.' and converts the consonant to lowercase.
 *          Algorithm: Iterative string manipulation with removal and insertion.
 *          Time Complexity: O(L * V), where L is the length of the string and V is
 *                           the number of vowels. String `erase` operations can be
 *                           O(L) in worst case.
 *          Domain: Competitive Programming.
 */
#include <algorithm>
#include <iostream>
#include <string.h>
#include <string>
using namespace std;

int main() {
  // Declare strings: 'str' to store the input, 's2' to build the output.
  string str, s2;
  // Read the input string.
  cin >> str;
  // Define an array of characters representing vowels (both uppercase and lowercase) to be removed.
  char charRemove[]{'A', 'O', 'Y', 'E', 'U', 'I', 'a', 'o', 'y', 'e', 'u', 'i'};
  
  // Calculate the number of vowels to remove.
  // Note: strlen on charRemove treats it as a C-style string, which might not be intended
  // if charRemove is not null-terminated. However, the current initialization makes it so.
  long n = strlen(charRemove);
  
  // Block Logic: Iterate through the list of vowels and remove each occurrence from the input string.
  // Invariant: After each iteration, one type of vowel (e.g., 'A' or 'a') is removed from 'str'.
  for (int j = 0; j < n; ++j) {
    // Uses std::remove to move all occurrences of charRemove[j] to the end of the string,
    // and then std::string::erase to remove them from the string.
    str.erase(remove(str.begin(), str.end(), charRemove[j]), str.end());
  }
  
  // Get the length of the modified string (after vowel removal).
  long long len = str.length();
  
  // Block Logic: Iterate through the modified string, prepend '.' to each character, and convert to lowercase.
  // Invariant: 's2' is built by adding '.' and the lowercase version of each consonant from 'str'.
  for (int j = 0; j < len; ++j) {
    s2 += '.';                  // Prepend a dot.
    s2 += tolower(str[j]);      // Append the lowercase version of the character.
  }
  // Output the final transformed string.
  cout << s2 << endl;
  
  // The standalone semicolon has no functional effect but is part of the original code.
  ;
  return 0; // Indicate successful program termination.
}