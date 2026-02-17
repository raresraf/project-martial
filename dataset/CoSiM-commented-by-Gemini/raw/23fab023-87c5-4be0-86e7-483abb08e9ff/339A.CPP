/*
 ~ Author    : @tridib_2003
 ~ Title     : 339A. Helpful Maths
 ~ Link      : https://codeforces.com/contest/339/problem/A
*/

/**
 * @file 339A.CPP
 * @brief Rearranges a string representing a sum of single-digit numbers into non-decreasing order.
 * @details The program takes a string like "3+2+1" and outputs "1+2+3". It achieves this
 * by extracting the numbers, sorting them, and then reconstructing the sum string.
 *
 * @algorithm Extraction and Sort.
 * @time_complexity O(N log N), where N is the number of digits. The dominant operation is sorting the extracted digits.
 * @space_complexity O(N) to store the string of extracted digits.
 */
#include <bits/stdc++.h>
using namespace std;

// Macro for long long integer type.
#define ll long long
#define PI 3.1415926535897932384626
#define MOD 1000000007
// Macro for creating a dynamic array.
#define vi vector<int>
#define mk(arr, n, type) type *arr = new type[n];
// Macro for a standard for-loop.
#define FOR(a, c) for (int(a) = 0; (a) < (c); (a)++)
// Macro for handling multiple test cases, not used in this specific solution.
#define w(x)                                                                   
  int x;
  cin >> x;
  while (x--)
// Macro to optimize C++ stream I/O for speed.
#define FIO                                                                    
  ios::sync_with_stdio(false);
  cin.tie(0);
  cout.tie(0);

int main() {

  FIO;

  string s;
  cin >> s;

  string ss;
  /**
   * Block Logic: Extracts all the digits (operands) from the input sum string.
   */
  for (int i = 0; i < s.length(); ++i) {
    if (s[i] != '+')
      ss += s[i];
  }

  // Sorts the string of digits. Because ASCII values for '1'-'9' are ordered,
  // this correctly sorts the numbers.
  sort(ss.begin(), ss.end());

  int pos = -1;
  /**
   * Block Logic: Reconstructs the original string by replacing the old digits
   * with the sorted digits, keeping the '+' operators in their original positions.
   */
  for (int i = 0; i < s.length(); ++i) {
    if (s[i] != '+')
      s[i] = ss[++pos];
  }

  cout << s << '\n';

  return 0;
}
