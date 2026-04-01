/**
 * @file 118A.CPP
 * @brief Solves the "String Task" problem from Codeforces (118A).
 * @details The program processes a string by deleting all vowels ('A', 'O', 'Y', 'E', 'U', 'I'),
 * inserting a '.' character before each consonant, and converting all uppercase consonants to lowercase.
 *
 * Algorithm: Character Transformation and Filtering.
 * 1. Read the input string.
 * 2. Convert the string into a vector of ASCII integer values (`p`).
 * 3. Iterate through the integer vector `p`.
 * 4. For each character, check if it is a vowel (case-insensitive) using its ASCII value. If so, ignore it.
 * 5. If it's a consonant, convert it to lowercase if it's uppercase.
 * 6. Prepend a '.' character (ASCII 46) to the consonant.
 * 7. Store the resulting '.' and lowercase consonant ASCII values in a new vector `q`.
 * 8. Finally, iterate through the result vector `q`, cast each integer back to a character, and print it.
 *
 * Time Complexity: O(L) where L is the length of the input string, due to multiple linear passes.
 * Space Complexity: O(L) to store the intermediate and final vectors of characters.
 */
#include <bits/stdc++.h>
using namespace std;

int main() {
  string a;
  cin >> a;
  int b, i, d, e, c = 0;
  // This vector 'p' will hold the ASCII values of the input string.
  vector<int> p;
  // First pass: Convert input string characters to their integer ASCII representations.
  for (i = 0; i < a.size(); i++) {
    b = a[i];
    p.push_back(b);
  }
  // This vector 'q' will hold the ASCII values of the resulting string.
  vector<int> q;
  char t;
  /**
   * Block Logic: Main processing loop to filter vowels and transform consonants.
   * It iterates over the ASCII values in vector 'p'.
   */
  for (i = 0; i < a.size(); i++) {
    /**
     * Pre-condition: Check if the character is a vowel (case-insensitive) using its ASCII value.
     * Vowels are 'A'(65), 'E'(69), 'I'(73), 'O'(79), 'U'(85), 'Y'(89) and their lowercase counterparts.
     */
    if (p[i] == 65 || p[i] == 69 || p[i] == 73 || p[i] == 79 || p[i] == 85 ||
        p[i] == 89 || p[i] == 97 || p[i] == 101 || p[i] == 105 || p[i] == 117 ||
        p[i] == 111 || p[i] == 121) {
      // If it's a vowel, do nothing.
      c = c + 0;
    } 
    /**
     * Pre-condition: Check if the character is an uppercase consonant (ASCII 65-90).
     */
    else if (p[i] <= 90 && p[i] >= 65) {
      // Convert uppercase to lowercase by adding 32 to its ASCII value.
      p[i] = p[i] + 32;
      // Prepend a '.' (ASCII 46).
      d = 46;
      q.push_back(d);
      q.push_back(p[i]);
      // Keep track of the size of the result vector 'q'.
      c = c + 2;
    } 
    /**
     * Logic: Handles lowercase consonants.
     */
    else {
      // Prepend a '.' (ASCII 46).
      d = 46;
      q.push_back(d);
      q.push_back(p[i]);
      // Keep track of the size of the result vector 'q'.
      c = c + 2;
    }
  }
  /**
   * Block Logic: Print the final transformed string.
   * Iterates through the result vector 'q', converts each ASCII value to a char, and prints.
   */
  for (i = 0; i < c; i++) {
    t = q[i];
    cout << t;
  }
}