/**
 * @file 118A.CPP
 * @brief Solution to Codeforces problem 118A - String Task.
 *        This program processes an input string by transforming it according to specific rules:
 *        1. All uppercase Latin letters are converted to their lowercase equivalents.
 *        2. All vowels ('a', 'o', 'y', 'e', 'u', 'i') are removed.
 *        3. A '.' character is inserted before each consonant.
 *        This implementation uses C-style char arrays and `std::vector<char>` for manipulation.
 *
 * Algorithm: Character-by-character Processing with Filtering and Prefixing
 *   1. Reads an input word into a C-style char array.
 *   2. Converts all characters of the word to lowercase and stores them in `myvector`.
 *   3. Iterates through `myvector`, removing all vowels. The loop index is adjusted
 *      after each removal to handle element shifting.
 *   4. Iterates through the modified `myvector` (now containing only consonants).
 *      For each consonant, it appends a '.' and then the consonant itself to `myvector2`.
 *   5. Prints the characters stored in `myvector2`.
 *
 * Time Complexity: O(L^2), where L is the length of the input string.
 *                  The vowel removal loop uses `vector::erase`, which can take O(L) time
 *                  for each removal, leading to an overall O(L^2) in the worst case.
 * Space Complexity: O(L) for storing characters in `myvector` and `myvector2`.
 */

#include <algorithm> // Required for std::tolower.
#include <cstring>   // Required for C-style string functions.
#include <ctype.h>   // Required for tolower().
#include <iostream>
#include <math.h> // Though included, not used in this specific problem.
#include <stdio.h>  // Required for printf, scanf (though not used directly here).
#include <string>   // Required for std::string (though primary input is char array).
#include <vector>   // Required for std::vector.
using namespace std;

int main() {
  /**
   * @brief Main function that solves the "String Task" problem.
   *
   * Reads a string, converts it to lowercase, removes vowels, and then
   * prints the remaining consonants prefixed with a dot.
   *
   * @return int Returns 0 upon successful execution.
   */
  char word[100]; // C-style character array to store the input word.
  int i = 0;      // Loop counter.
  vector<char> myvector;  // Vector to store lowercase characters and then consonants.
  vector<char> myvector2; // Vector to store the final output with '.' prefixes.
  cin >> word;            // Reads the input word.
  /**
   * Block Logic: Converts each character of the input `word` to lowercase and stores it in `myvector`.
   * Invariant: `myvector` contains the lowercase representation of `word` up to `word[i-1]`.
   * Pre-condition: `word` is a valid C-style string.
   */
  while (word[i] != '\0') {
    myvector.push_back(tolower(word[i])); // Converts to lowercase and appends to vector.
    i++;
  }
  /**
   * Block Logic: Iterates through `myvector` and removes all vowel characters.
   * Invariant: `myvector` is being modified in-place to contain only consonants.
   * Pre-condition: `myvector` contains lowercase characters.
   */
  for (i = 0; i < myvector.size(); i++) {
    // Conditional Logic: Checks if the current character is a vowel.
    if (myvector[i] == 'a' || myvector[i] == 'o' || myvector[i] == 'i' ||
        myvector[i] == 'y' || myvector[i] == 'e' || myvector[i] == 'u') {
      myvector.erase(myvector.begin() + i); // Inline: Removes the vowel character.
      i--; // Inline: Decrements `i` to re-check the element that shifted into this position.
    }
  }

  /**
   * Block Logic: Iterates through the consonant-only `myvector` and constructs
   *              `myvector2` by adding a '.' prefix to each consonant.
   * Invariant: `myvector2` grows with '.' followed by each consonant from `myvector`.
   * Pre-condition: `myvector` contains only lowercase consonants.
   */
  for (i = 0; i < myvector.size(); i++) {
    myvector2.push_back('.'); // Adds the '.' prefix.
    myvector2.push_back(myvector[i]); // Adds the consonant.
  }
  /**
   * Block Logic: Prints all characters from `myvector2` to the console.
   * Pre-condition: `myvector2` contains the fully transformed string.
   */
  for (i = 0; i < myvector2.size(); i++) {
    cout << myvector2[i]; // Prints each character.
  }
  return 0;
}
