/**
 * @file 339A.CPP
 * @brief Solves the "Helpful Maths" problem from Codeforces (Problem 339A).
 * @details The program reorders a string representing a sum of digits (1, 2, 3)
 * into non-decreasing order. For example, "3+2+1" becomes "1+2+3".
 *
 * @algorithm
 * This solution uses a counting sort approach, which is efficient given the
 * small and fixed range of numbers.
 * 1. Read the input string (e.g., "3+2+1").
 * 2. Count the occurrences of each digit ('1', '2', '3') in the string.
 * 3. Build a new string by appending "1+" for each '1' counted, followed by
 *    "2+" for each '2', and "3+" for each '3'. This results in a sorted
 *    string with a trailing plus (e.g., "1+2+3+").
 * 4. Remove the trailing "+" from the constructed string.
 * 5. Print the final, sorted string.
 *
 * @time_complexity O(L), where L is the length of the input string, as it involves
 * a few linear passes over the string and the counts.
 * @space_complexity O(L) to store the new string.
 */
// https://codeforces.com/contest/339/problem/A

/* https://codeforces.com/problemset/problem/339/A */

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
   * @block_logic Count the frequency of each digit ('1', '2', '3').
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
   * @block_logic Reconstruct the string in sorted order based on the counts.
   * Each number is appended with a '+' sign.
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

  // Remove the trailing '+' from the end of the reconstructed string.
  string s3 = s2.substr(0, s2.size() - 1);

  cout << s3;
}