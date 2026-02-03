/**
 * @file 71A.CPP
 * @brief Solution to the "Way Too Long Words" problem.
 * @see https://codeforces.com/problemset/problem/71/A
 *
 * This program abbreviates long words. If a word is longer than 10
 * characters, it is replaced by its first character, the number of characters
 * between the first and last characters, and its last character.
 *
 * Algorithm:
 * 1. Read the number of words to process (n).
 * 2. Loop n times:
 *    a. Read a word.
 *    b. If the word's length is greater than 10, print the abbreviation.
 *    c. Otherwise, print the word as is.
 *
 * Time Complexity: O(N * L), where N is the number of words and L is the
 * maximum length of a word (though string length calculation is typically O(1)
 * for std::string). The dominant factor is reading the input strings.
 * Space Complexity: O(L) to store the longest word.
 */
#include <cstdio>
#include <iostream>
#include <string.h>

using namespace std;

int main()

{

  int n, l;

  cin >> n;

  string s;

  // Invariant: The loop processes n words, applying the abbreviation rule to each.
  while (n--)

  {

    cin >> s;

    l = s.length();

    // Block Logic: Determines whether to abbreviate the word based on its length.
    if (l <= 10)

    {
      cout << s << endl;
    }

    else

    {

      cout << s[0] << l - 2 << s[l - 1] << endl;
    }
  }

  return 0;
}