/**
 * @68f87659-c53f-4404-b350-9c24f43c1c96/71A.CPP
 * @brief This program solves Codeforces problem 71A, "Way Too Long Words".
 * It processes a list of words, abbreviating those longer than 10 characters.
 * Abbreviation involves replacing the middle part of the word with its length minus two,
 * e.g., "localization" (length 12) becomes "l10n". Words 10 characters or shorter
 * are printed as they are.
 *
 * Algorithm: Iterative processing of words with conditional abbreviation based on length.
 * String manipulation is used to construct the abbreviated form.
 * Time Complexity: O(N * L) where N is the number of words and L is the average length of a word.
 * Space Complexity: O(L_max) for storing the longest word.
 */
#include <bits/stdc++.h> // Includes many standard C++ library headers, including <iostream>, <string>, and <cstdio>.
using namespace std;     // Uses the standard namespace.

/**
 * @brief Main function to process words for abbreviation.
 * Reads the number of words, then reads each word, and prints either
 * the original or abbreviated version based on its length.
 * @return 0 implicitly, as there is no explicit return statement at the end of main.
 */
int main() {
  int n; // Declares 'n', representing the number of words to process.
  string in; // Declares a string variable to store the input word.
  
  scanf("%d", &n); // Reads 'n' (number of words) from standard input.
  
  /**
   * Block Logic: Processes 'n' words. Each word is either printed as is or abbreviated based on its length.
   * Invariant: Each word from input is processed and printed according to the rules.
   */
  while (n--) { // Loops 'n' times, decrementing 'n' in each iteration.
    cin >> in; // Reads a word into `in`.
    
    /**
     * Block Logic: Determines whether to print the word as is or its abbreviation.
     * Pre-condition: `in.size()` holds the length of the current word.
     * Invariant: The word is printed in its original form if length <= 10, abbreviated otherwise.
     */
    if (in.size() > 10) { // If the word length is greater than 10.
      int len = in.size() - 2; // Calculates the middle part of the abbreviation (length - 2).
      // Reconstructs the string: first char + string representation of length - 2 + last char.
      in = in[0] + to_string(len) + in.back(); 
    }
    cout << in << endl; // Prints the word (original or abbreviated), followed by a newline.
  }
}
