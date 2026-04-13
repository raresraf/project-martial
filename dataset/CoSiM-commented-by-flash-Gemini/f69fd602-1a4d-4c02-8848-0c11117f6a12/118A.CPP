/**
 * @file 118A.CPP
 * @brief This program transforms an input string by converting all uppercase
 *        consonants to lowercase and prepending a '.' before each consonant.
 *        Vowels (A, E, I, O, U, Y, a, e, i, o, u, y) are excluded from the output.
 * Algorithm: String processing, character classification, and transformation.
 * Time Complexity: O(L) where L is the length of the input string, due to linear
 *                  traversals for character processing and output generation.
 * Space Complexity: O(L) for storing intermediate character lists (vectors 'p' and 'q').
 */
#include <bits/stdc++.h>
using namespace std;

int main() {
  string a; // Input string.
  // Reads the input string.
  cin >> a;
  // b: temporary variable for character ASCII value.
  // i: loop counter.
  // d, e: unused variables.
  // c: count of characters in the transformed string.
  int b, i, d, e, c = 0;
  // Vector to store ASCII values of characters from the input string.
  vector<int> p;
  /**
   * Block Logic: Copies ASCII values of characters from the input string 'a' to vector 'p'.
   *              This loop is functionally redundant as 'a[i]' could be used directly.
   */
  for (i = 0; i < a.size(); i++) {
    b = a[i];
    p.push_back(b);
  }
  // Vector to store ASCII values of characters for the transformed output string.
  vector<int> q;
  char t; // Temporary character for output.
  /**
   * Block Logic: Iterates through the stored character ASCII values to perform string transformation.
   *              - Vowels are skipped.
   *              - Consonants are converted to lowercase (if uppercase) and prepended with a '.'.
   * Invariant: After each iteration, a character from the input string has been processed,
   *            and corresponding transformed characters (if any) are added to vector 'q'.
   */
  for (i = 0; i < a.size(); i++) {
    // Checks if the character is a vowel (case-insensitive).
    // ASCII values: A=65, E=69, I=73, O=79, U=85, Y=89
    //               a=97, e=101, i=105, o=111, u=117, y=121
    if (p[i] == 65 || p[i] == 69 || p[i] == 73 || p[i] == 79 || p[i] == 85 ||
        p[i] == 89 || p[i] == 97 || p[i] == 101 || p[i] == 105 || p[i] == 117 ||
        p[i] == 111 || p[i] == 121) {
      c = c + 0; // Vowels are ignored, 'c' remains unchanged.
    }
    // Checks if the character is an uppercase consonant.
    else if (p[i] <= 90 && p[i] >= 65) {
      p[i] = p[i] + 32; // Converts uppercase consonant to lowercase (e.g., 'A' to 'a').
      d = 46;           // ASCII value for '.'
      q.push_back(d);   // Appends '.' to the transformed string.
      q.push_back(p[i]); // Appends the lowercase consonant.
      c = c + 2;        // Increments 'c' by 2 for '.' and the consonant.
    }
    // Handles lowercase consonants and other non-vowel characters.
    else {
      d = 46;           // ASCII value for '.'
      q.push_back(d);   // Appends '.' to the transformed string.
      q.push_back(p[i]); // Appends the consonant.
      c = c + 2;        // Increments 'c' by 2 for '.' and the consonant.
    }
  }
  /**
   * Block Logic: Prints the transformed string stored in vector 'q'.
   * Invariant: Each character from 'q' is printed sequentially.
   */
  for (i = 0; i < c; i++) {
    t = q[i]; // Casts the ASCII value back to a character.
    cout << t; // Prints the character.
  }
}