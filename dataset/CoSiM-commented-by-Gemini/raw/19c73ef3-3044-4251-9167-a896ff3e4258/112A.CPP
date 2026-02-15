/**
 * @file 112A.CPP
 * @brief Solution to the "Petya and Strings" competitive programming problem.
 * @details This program performs a case-insensitive lexicographical comparison of two strings.
 * It compares the strings character by character and terminates at the first point of difference.
 *
 * @section algorithm Algorithm
 * The algorithm normalizes and compares the strings on a per-character basis:
 * 1. Read the two input strings, `p` and `s`.
 * 2. Loop through the strings from the first character to the last.
 * 3. In each iteration, for both strings, convert the current character to a
 *    common alphabetical index (0-25).
 *    - This is done by checking if the character's ASCII value falls within the
 *      uppercase range (65-90 for 'A'-'Z') or the lowercase range (97-122 for 'a'-'z')
 *      and then subtracting the corresponding base value (65 for 'A' or 97 for 'a').
 * 4. Compare these two normalized index values (`val1` and `val2`).
 * 5. If they differ, a lexicographical difference has been found. The program prints -1 or 1
 *    accordingly and terminates immediately.
 * 6. If the loop completes without any differences being found, the strings are
 *    considered equal, and the program prints 0.
 *
 * @section complexity Complexity Analysis
 * - Time Complexity: O(L), where L is the length of the strings. In the best case
 *   (a difference at the first character), it's O(1). In the worst case (equal strings),
 *   it iterates through the entire length.
 * - Space Complexity: O(L) to store the two input strings.
 */
#include <bits/stdc++.h>
using namespace std;
int main() {
  string p, s;
  int n, i, val1, val2;
  cin >> p >> s;
  n = p.length();

  // Iterate through the strings character by character for comparison.
  for (i = 0; i < n; i++) {
    // Normalize the character from the first string 'p' to an index from 0 to 25.
    if (p[i] >= 65 && p[i] <= 90) // ASCII for 'A' is 65, 'Z' is 90.
      val1 = p[i] - 65;
    else // ASCII for 'a' is 97.
      val1 = p[i] - 97;
    
    // Normalize the character from the second string 's'.
    if (s[i] >= 65 && s[i] <= 90)
      val2 = s[i] - 65;
    else
      val2 = s[i] - 97;
    
    // Compare the normalized values and terminate at the first difference.
    if (val1 < val2) {
      cout << "-1";
      return 0; // Terminate early as the lexicographical order is determined.
    }
    if (val1 > val2) {
      cout << "1";
      return 0; // Terminate early.
    }
  }
  
  // If the loop completes, no differences were found, so the strings are equal.
  cout << "0";
  return 0;
}
