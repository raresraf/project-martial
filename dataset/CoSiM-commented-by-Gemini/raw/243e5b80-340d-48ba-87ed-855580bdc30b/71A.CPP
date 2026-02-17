/**
 * @file 71A.CPP
 * @brief Solution to the "Way Too Long Words" problem (variant implementation).
 * @details This program abbreviates words that are longer than 10 characters. The abbreviation
 * consists of the first character, the number of internal characters, and the last character.
 * This implementation uses a mix of C-style (`scanf`) and C++-style (`cin`) I/O.
 *
 * @algorithm String Abbreviation
 * @complexity
 * Time Complexity: O(T * L), where T is the number of test cases and L is the maximum length of a word.
 * The complexity is determined by the input operation that reads T words of up to length L.
 * Space Complexity: O(L), for storing the longest input word.
 */
#include <bits/stdc++.h>
using namespace std;
int main() {
  int n;
  // Read the number of test cases. The `\n` in the format string consumes the trailing newline.
  scanf("%d\n", &n);

  // Block Logic: Loop through each of the 'n' test cases.
  while (n--) {
    string word;
    cin >> word;

    // Block Invariant: Check if the word's length is strictly greater than 10.
    if (word.length() > 10)
      // Logic: If the word is too long, print its abbreviation.
      // The abbreviation is formed by the first character, the number of middle characters (length - 2),
      // and the last character.
      cout << word[0] << word.length() - 2 << word[word.length() - 1];
    else
      // If the word is 10 characters or shorter, print it as is.
      cout << word;

    cout << endl;
  }
}