/**
 * @file 339A.CPP
 * @brief Rearranges a string-represented sum of numbers into non-decreasing order.
 * @details This program solves the Codeforces problem 339A. It parses a string
 * containing a sum of single-digit numbers (e.g., "3+2+1"), sorts the numbers,
 * and then prints the sum with the numbers in non-decreasing order (e.g., "1+2+3").
 *
 * @section
 * Algorithm:
 * 1. Read the input string which represents the sum.
 * 2. Extract all the numeric digits from the string, ignoring the '+' symbols.
 * 3. Store these digits in a character vector.
 * 4. Sort the vector of digits in ascending order.
 * 5. Reconstruct the sum string by concatenating the sorted digits with '+' symbols in between.
 *
 * @section
 * Complexity Analysis:
 * - Time Complexity: O(N log N), where N is the length of the input string. This is dominated by the sorting operation on the digits.
 * - Space Complexity: O(N), for storing the digits in a vector and for the answer string.
 */

/*
    Problem Link    :   https://codeforces.com/problemset/problem/339/A
    Contest Link    :   https://codeforces.com/contest/339/problem/A
    Status          :   Accepted

    Author          :   Nikhil Nagrale
    Codeforces      :   https://codeforces.com/profile/nikhilnagrale2
    Codechef        :   https://www.codechef.com/users/nikhilnagrale2
    Github          :   https://github.com/nikhilnagrale2

    Copyright       :   Solve it on your own.
*/

#include <bits/stdc++.h>
using namespace std;

int main() {
  string s;
  cin >> s;
  vector<char> v;
  
  // Block Logic: Parse the input string and extract the digits.
  // Pre-condition: The input string `s` is a sum of single digits separated by '+'.
  for (char x : s) {
    if (x != '+')
      v.push_back(x);
  }

  // Algorithm Step: Sort the extracted digits to place them in non-decreasing order.
  sort(v.begin(), v.end());

  string ans = "";
  ans += v[0];
  v.erase(v.begin());

  // Block Logic: Reconstruct the sorted sum string.
  // Invariant: The `ans` string is built by appending '+' and the next smallest digit in each iteration.
  for (char x : v) {
    ans += '+';
    ans += x;
  }
  
  // Output the final rearranged sum.
  cout << ans << endl;
  return 0;
}