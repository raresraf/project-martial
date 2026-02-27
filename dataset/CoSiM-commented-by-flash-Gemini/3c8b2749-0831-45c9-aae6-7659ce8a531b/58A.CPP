/**
 * @file 58A.CPP
 * @brief Solution for competitive programming problem: checks if "hello" is a subsequence of an input string.
 *
 * @details This program implements a two-pointer approach to efficiently determine
 *          if the target string "hello" can be formed by deleting zero or more
 *          characters from the input string without changing the order of the
 *          remaining characters.
 *          Algorithm: Two-pointer subsequence matching.
 *          Time Complexity: O(L), where L is the length of the input string, as
 *                           each character is visited at most once.
 *          Domain: Competitive Programming.
 */
#include <bits/stdc++.h>
#include <cstring>
using namespace std;

int main() {
  // Declare character arrays to store the input string and the target subsequence "hello".
  char str[101], str2[10];
  // Declare integer variables for loop counters, string lengths. 'j' serves as a pointer into the input string,
  // while 'i' is the main loop counter, intended to iterate through the input string and also
  // mistakenly used to index the pattern string 'str2'. 'count' tracks matched characters of "hello",
  // and 'flag' indicates if "hello" has been found.
  int i, j = 0, k, l, m, n, count = 0, flag = 0;

  // Read the input string from standard input.
  cin >> str;
  // Calculate the length of the input string, stored in 'l'.
  l = strlen(str);

  // Initialize the target string to find as a subsequence.
  strcpy(str2, "hello");

  // Block Logic: Attempts to find "hello" as a subsequence within the input string 'str'.
  // This loop contains unconventional logic and a potential out-of-bounds access on `str2`.
  // Invariant: 'i' iterates through the input string 'str'. 'j' points to a position in 'str'.
  // 'count' tracks the number of characters of "hello" found so far.
  for (i = 0; i < l; i++) {
    // Precondition: Checks for a match between a character from `str2` (indexed by 'i')
    // and a character from `str` (indexed by 'j').
    // Note: Indexing `str2` with `i` can lead to out-of-bounds access if `i` >= 5.
    if (str2[i] == str[j]) {
      j++;     // Advance the 'j' pointer within `str`.
      count++; // Increment the count of matched characters.

      // Precondition: If all 5 characters of "hello" have been matched.
      if (count == 5) {
        flag = 1; // Set flag to indicate successful subsequence match.
        break;    // Exit the loop.
      }
    } else {
      // Unconventional Logic: If no match, rewind 'i' and advance 'j'.
      // This logic means the current character `str[i]` will be re-evaluated
      // in the next iteration against a character from `str2` at `i+1`.
      // Simultaneously, `j` advances, effectively skipping a character in `str`
      // that was previously considered, making the subsequence logic unusual.
      i--;
      j++;
    }

    // Precondition: If the 'j' pointer has reached the end of the input string 'str',
    // the search cannot continue.
    if (j == l)
      break; // Exit the loop.
  }

  // Block Logic: Output "YES" if the 'flag' indicates "hello" was found as a subsequence, "NO" otherwise.
  if (flag == 1)
    cout << "YES" << endl;
  else
    cout << "NO" << endl;
}
