/**
 * @file 339A.CPP
 * @brief Solution for the "Helpful Maths" problem (Codeforces 339A).
 * @details This program rearranges a string representing a sum of numbers (1, 2, and 3)
 * into non-decreasing order. It parses the digit characters into an integer
 * vector, sorts the vector, and then prints the elements separated by plus signs.
 *
 * @algorithm Character-by-character parsing, sorting, and formatted printing.
 * @time_complexity O(L + N log N), where L is the length of the input string and N
 * is the number of summands. The complexity is dominated by the parsing and sorting steps.
 * @space_complexity O(N) to store the vector of integers.
 */
#include <bits/stdc++.h>
#include <string.h>
using namespace std;
// Macro for fast I/O operations in C++.
#define IOS                                                                    \
  ios::sync_with_stdio(0); \
  cin.tie(0); \
  cout.tie(0);
#define endl "\n"
#define Max(x, y, z) max(x, max(y, z))
#define Min(x, y, z) min(x, min(y, z))
// A common competitive programming practice to prevent integer overflow.
#define int long long
// Macros for debugging output to cerr.
#define trace1(x) cerr << #x << ": " << x << endl
#define trace2(x, y) cerr << #x << ": " << x << " | " << #y << ": " << y << endl
#define trace3(x, y, z) \
  cerr << #x << ":" << x << " | " << #y << ": " << y << " | " << #z << ": " \
       << z << endl

int32_t main() {
  IOS;
  string s;
  vector<int> v;
  cin >> s;

  /**
   * Block Logic: Parses the input string character by character. If a character
   * is a digit ('1', '2', or '3'), its integer equivalent is added to a vector.
   * This effectively extracts the numbers from the sum.
   */
  for (int i = 0; i < s.length(); ++i) {
    if (s[i] == '1')
      v.push_back(1);
    else if (s[i] == '2')
      v.push_back(2);
    else if (s[i] == '3')
      v.push_back(3);
  }
  
  // Sorts the vector of integers in non-decreasing order.
  sort(v.begin(), v.end());

  // Block Logic: Prints the sorted numbers, separated by '+' signs.
  cout << v[0];
  for (int i = 1; i < v.size(); ++i) {
    cout << "+" << v[i];
  }
  return 0;
}
