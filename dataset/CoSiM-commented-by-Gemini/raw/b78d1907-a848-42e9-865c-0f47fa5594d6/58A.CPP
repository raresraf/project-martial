/**
 * @file 58A.CPP
 * @brief An inefficient solution to the "Chat room" problem (Codeforces 58A).
 * @link https://codeforces.com/contest/58/problem/A
 *
 * @details
 * This program checks if a given string `s` contains the word "hello" as a subsequence.
 * A subsequence's characters must appear in the same order but not necessarily
 * consecutively.
 *
 * This solution uses a brute-force approach with five nested `for` loops. Each loop
 * searches for the next character of the word "hello" in the substring that follows
 * the previously found character. If all five characters are found in order, it
 * prints "YES" and exits.
 *
 * @warning This implementation is highly inefficient due to its nested loop structure.
 * The time complexity is polynomial, potentially up to O(L^5) in the worst case,
 * where L is the length of the string. This is far from the optimal O(L) greedy
 * solution and would be too slow for larger constraints.
 *
 * Algorithm: Brute-Force Nested Search
 * Time Complexity: O(L^5)
 * Space Complexity: O(L)
 */
#include <iostream>
#include <bits/stdc++.h> // Includes <string> and other headers
using namespace std;

int main()

{
  int n;
  string s;
  cin >> s;
  n = s.length();

  /**
   * @brief Brute-force search for the subsequence "hello".
   * @warning This nested loop structure has a worst-case time complexity of O(L^5)
   * and is extremely inefficient. An O(L) greedy approach is optimal.
   *
   * @details Each level of nesting searches for the next character of "hello"
   * sequentially through the remainder of the string.
   */
  for (int i = 0; i < n; i++)
  {
    // Search for 'h'
    if (s[i] == 'h')
    {
      // If 'h' is found, search for 'e' after it
      for (int j = i + 1; j < n; j++)
      {
        if (s[j] == 'e')
        {
          // If 'e' is found, search for 'l' after it
          for (int k = j + 1; k < n; k++)
          {
            if (s[k] == 'l')
            {
              // If first 'l' is found, search for second 'l'
              for (int l = k + 1; l < n; l++)
              {
                if (s[l] == 'l')
                {
                  // If second 'l' is found, search for 'o'
                  for (int m = l + 1; m < n; m++)
                  {
                    if (s[m] == 'o')
                    {
                      // "hello" found as a subsequence.
                      cout << "YES" << endl;
                      return 0; // Terminate successfully.
                    }
                  }
                }
              }
            }
          }
        }
      }
    }
  }

  // If the loops complete without finding "hello", it is not a subsequence.
  cout << "NO" << endl;
}