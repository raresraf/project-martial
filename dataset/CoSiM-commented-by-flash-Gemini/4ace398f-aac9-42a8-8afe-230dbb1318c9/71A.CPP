/**
 * @4ace398f-aac9-42a8-8afe-230dbb1318c9/71A.CPP
 * @brief This program abbreviates words that are longer than 10 characters,
 *        by printing the first letter, the count of intermediate characters,
 *        and the last letter. Shorter words are printed as is.
 * Algorithm: String length check and conditional printing based on length.
 * Time Complexity: O(N * L) where N is the number of words and L is the
 *                  average length of a word, due to string length calculation and character access.
 * Space Complexity: O(L) for storing each input word.
 */
#include <iostream>
using namespace std;
int main() {
  int n; // Declare 'n' to store the number of words to process.
  string s; // Declare a string 's' to store each input word.
  // Functional Utility: Read the number of words.
  cin >> n;
  // Block Logic: Iterate 'n' times, processing each word.
  while (n--) {
    // Functional Utility: Read the current word.
    cin >> s;
    // Pre-condition: Check if the word's length is greater than 10.
    if (s.length() > 10) {
      // Action: Print the abbreviated form: first character, length-2, last character.
      cout << s[0] << s.length() - 2 << s[s.length() - 1] << endl;
    } else {
      // Action: Otherwise, print the word as is.
      cout << s << endl;
    }
  }
  return 0; // Indicate successful program termination.
}