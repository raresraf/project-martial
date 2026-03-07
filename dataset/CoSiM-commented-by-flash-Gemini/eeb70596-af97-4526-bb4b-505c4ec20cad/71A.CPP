/**
 * @file 71A.CPP
 * @brief Solution to a competitive programming problem that abbreviates long words.
 * The program reads an integer 'n' followed by 'n' words. For each word, if its length
 * is greater than 10 characters, it is abbreviated to its first character, the count
 * of intermediate characters, and its last character (e.g., "localization" becomes "l10n").
 * Otherwise, the word is printed as is.
 * Algorithm: String length-based conditional abbreviation.
 * Time Complexity: O(N * L), where N is the number of words and L is the maximum length of a word.
 * Space Complexity: O(L) for storing the input string.
 */
#include <bits/stdc++.h>

using namespace std;

int main() {
  // Declares an integer 'n' to store the total number of words.
  int n;
  // Declares a string 'inp' to store each input word.
  string inp;
  
  // Reads the total number of words from standard input using scanf.
  scanf("%d", &n);
  // Consumes the remaining newline character after scanf to prevent issues with subsequent getline.
  getline(cin, inp); 
  
  /**
   * Block Logic: Iterates 'n' times to process each word.
   * Invariant: At the start of each iteration, 'i' represents the index of the current word being processed.
   */
  for (int i = 0; i < n; i++) {
    // Reads an entire line (word) from standard input.
    cin >> inp;
    
    /**
     * Conditional Logic: Checks if the length of the current word exceeds 10 characters.
     */
    if (inp.length() > 10) {
      // If the word is too long, prints its abbreviation: first character, length-2, last character.
      cout << inp[0] << inp.length() - 2 << inp[inp.length() - 1] << endl;
    } else {
      // Otherwise, prints the word as is.
      cout << inp << endl;
    }
  }
  
  return 0;
}