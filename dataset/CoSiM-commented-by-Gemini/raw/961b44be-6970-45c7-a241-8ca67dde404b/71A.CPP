/**
 * @file 71A.CPP
 * @brief Abbreviates words that are longer than 10 characters.
 *
 * This program solves the "Way Too Long Words" problem from Codeforces. It first
 * reads all input words into an array and then iterates through the array to
 * print either the original word or its abbreviation if the word's length
 * is greater than 10.
 *
 * @algorithm Batch Processing / String Manipulation
 * @complexity
 * - Time: O(L_total), where L_total is the sum of the lengths of all words,
 *   as the program involves reading all strings and then iterating through them again.
 * - Space: O(L_total), as it stores all `n` words in an array in memory before processing.
 */
#include <bits/stdc++.h>
#include <iostream>
using namespace std;
int main() {
  int n;
  // An array to store all the words. The size 100 is based on problem constraints.
  string s[100];
  cin >> n;
  
  /**
   * @brief Block Logic: Reads all `n` words into an array.
   * This is a batch-processing approach where all input is collected first.
   */
  for (int i = 0; i < n; i++) {
    cin >> s[i];
  }

  /**
   * @brief Block Logic: Iterates through the stored words to process and print them.
   */
  for (int i = 0; i < n; i++) {
    // Pre-condition: Check if the length of the current word exceeds 10.
    if (s[i].length() > 10)
      /**
       * @brief Logic: Constructs and prints the abbreviation.
       * The format is the first character, the number of middle characters
       * (length - 2), and the last character.
       */
      cout << s[i][0] << s[i].length() - 2 << s[i][s[i].length() - 1] << endl;
    else
      // If the word is not too long, print it as is.
      cout << s[i] << endl;
  }
  return 0;
}