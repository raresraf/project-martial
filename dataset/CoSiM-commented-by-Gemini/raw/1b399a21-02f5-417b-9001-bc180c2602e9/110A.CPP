/**
 * @file 110A.CPP
 * @brief Solution for a competitive programming problem involving "nearly lucky numbers".
 *
 * This program determines if a given number is "nearly lucky". A number is
 * considered nearly lucky if the total count of its "lucky digits" ('4' and '7')
 * is itself a lucky number (either 4 or 7).
 *
 * @algorithm
 * 1. Read the input number as a string.
 * 2. Count the occurrences of the characters '4' and '7' in the string.
 * 3. Check if the resulting count is equal to 4 or 7.
 * 4. Print "YES" if it is, "NO" otherwise.
 *
 * @time_complexity O(L), where L is the number of digits in the input number.
 * @space_complexity O(L) to store the input string.
 */
#include <bits/stdc++.h>
using ll = long long;
using namespace std;

int main() {
  // Functional Utility: Decouples C++ streams from standard C streams
  // to optimize high-throughput I/O for competitive programming.
  ios::sync_with_stdio(false);
  string str;
  cin >> str;

  // Block Logic: Count the number of "lucky digits" ('4' or '7')
  // present in the input string.
  int cnt = count_if(str.begin(), str.end(),
                     [](char a) { return a == '4' || a == '7'; });

  // Block Logic: Determine if the count of lucky digits is itself a lucky number.
  // Pre-condition: 'cnt' holds the total number of '4's and '7's.
  // Invariant: The program outputs "YES" if the count is 4 or 7, "NO" otherwise.
  if (cnt == 4 || cnt == 7)
    cout << "YES" << endl;
  else
    cout << "NO" << endl;
  return 0;
}
