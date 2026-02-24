/**
 * @file 71A.CPP
 * @brief Solution for Codeforces Problem 71A - Way Too Long Words.
 *
 * This program takes a list of words and abbreviates any word that is
 * strictly longer than 10 characters. The abbreviation consists of the
 * first letter of the word, followed by the number of characters between
 * the first and last letters, and then the last letter of the word.
 * Words 10 characters or shorter are printed as is.
 *
 * Algorithm: String length check and conditional formatting. For each word,
 * its length is checked. If it exceeds 10, it's abbreviated; otherwise, it's printed whole.
 * Time Complexity: O(N * L), where N is the number of words and L is the average
 *                  length of a word (as string operations like size() are O(L)).
 * Space Complexity: O(L_max) for storing the longest word.
 */

#include <bits/stdc++.h> // Includes various standard libraries, e.g., iostream, string.
using namespace std;     // Uses the standard namespace to avoid prefixing std::.

int main() {
  /**
   * @brief Main function to execute the Way Too Long Words problem logic.
   *
   * Reads an integer 'n' representing the number of words to process.
   * Then, it enters a loop to read 'n' words one by one. For each word,
   * it applies the abbreviation logic and prints the result to standard output.
   */
  // Declare 'n' to store the number of words.
  int n;
  // Read the total number of words to be processed.
  cin >> n;

  // Block Logic: Process each word.
  // Invariant: The loop iterates 'n' times, processing one word in each iteration.
  for (int i = 0; i < n; ++i) {
    // Declare 's' to store the current word.
    string s;
    // Read the current word.
    cin >> s;
    // Conditional Logic: Check if the word needs abbreviation.
    // Pre-condition: The string 's' contains the current word.
    if (s.size() > 10) {
      // Calculate the number of characters between the first and last.
      // s.size() gives total length, subtract 2 for first and last characters.
      int a = s.size() - 2;
      // Print the abbreviated word using C-style printf for formatted output.
      // %c for characters, %d for integer.
      printf("%c%d%c\n", s[0], a, s[s.size() - 1]);
    } else
      // If the word is 10 characters or shorter, print it as is, followed by a newline.
      cout << s << endl;
  }

  return 0; // Indicate successful program execution.
}

// codeforces.com/problemset/problem/71/A