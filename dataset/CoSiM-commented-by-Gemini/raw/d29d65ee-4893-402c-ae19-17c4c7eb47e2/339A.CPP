// https://codeforces.com/contest/339/problem/A
/* https://codeforces.com/problemset/problem/339/A */

/**
 * @file 339A.CPP
 * @brief Solves the "Helpful Maths" problem (Codeforces 339A).
 * @details The program takes a string representing a sum of numbers (1, 2, and 3),
 * and reorders it into a non-decreasing sequence. For example, "3+2+1"
 * becomes "1+2+3".
 *
 * @section algorithm
 * Algorithm: Counting Sort
 * 1. The input string is read.
 * 2. The code iterates through the string, counting the occurrences of the
 *    characters '1', '2', and '3'. The '+' symbols are ignored.
 * 3. A new string is constructed by first appending all the "1+"s, then all
 *    the "2+"s, and finally all the "3+"s, based on the counts.
 * 4. The final, trailing '+' is removed from the newly constructed string.
 * 5. The resulting sorted string is printed.
 *
 * @section complexity
 * - Time Complexity: O(L), where L is the length of the input string. This is
 *   because the code makes a few linear passes over the string or its elements.
 * - Space Complexity: O(L), as a new string is constructed whose length is
 *   proportional to the original string's length.
 */
#include <iostream>
#include <string>
#include <bits/stdc++.h>

using namespace std;

int main()
{
  string s1;
  cin >> s1;
  int one = 0, two = 0, three = 0;

  /**
   * @brief This loop serves as a frequency counter.
   * It iterates through the input string `s1` and counts how many '1's, '2's,
   * and '3's are present, ignoring the '+' characters.
   */
  for (int i = 0; i < s1.length(); i++)
  {
    if (s1[i] == '1')
      one++;
    if (s1[i] == '2')
      two++;
    if (s1[i] == '3')
      three++;
  }

  string s2 = "";

  /**
   * @brief This section reconstructs the string in sorted order.
   * Based on the counts, it builds a new string `s2` by appending the
   * numbers in non-decreasing order, each followed by a '+'.
   */
  if (one > 0)
  {
    for (int i = 0; i < one; i++)
    {
      s2 = s2 + "1+";
    }
  }
  if (two > 0)
  {
    for (int i = 0; i < two; i++)
    {
      s2 = s2 + "2+";
    }
  }
  if (three > 0)
  {
    for (int i = 0; i < three; i++)
    {
      s2 = s2 + "3+";
    }
  }

  /**
   * @brief Removes the trailing '+' from the reconstructed string.
   * The construction logic leaves an extra '+' at the end, which is trimmed here
   * before printing the final result.
   */
  string s3 = s2.substr(0, s2.size() - 1);
  cout << s3;
}