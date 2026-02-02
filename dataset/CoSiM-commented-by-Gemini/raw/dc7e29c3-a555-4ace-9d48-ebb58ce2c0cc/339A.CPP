/**
 * @file 339A.CPP
 * @brief Solves the "Helpful Maths" problem from Codeforces.
 * @details The program takes a string representing a sum of single-digit numbers
 * (1, 2, or 3) and rearranges them into non-decreasing order.
 *
 * Algorithm:
 * 1. Read the input string which is a sum, e.g., "3+2+1".
 * 2. The numbers are at even indices (0, 2, 4, ...).
 * 3. A simple bubble sort is applied to the characters at these even indices.
 * 4. The nested loops compare each number with every other number and swap them
 *    if they are in the wrong order.
 * 5. Finally, the modified string with sorted numbers is printed.
 *
 * Time Complexity: O(N^2), where N is the number of digits in the sum. This is due
 * to the bubble sort implementation. Given the constraints on the input size in
 * the problem, this is acceptable.
 * Space Complexity: O(L), where L is the length of the input string.
 */
#include <bits/stdc++.h>
using namespace std;

#define ll long long int
#define mp make_pair
#define pb push_back
#define fi first
#define si second
#define fastIO                                                                 \
  ios_base::sync_with_stdio(false);                                            \
  cin.tie(NULL);
#define F(a, n)                                                                \
  for (int i = 0; i < n; i++) {                                                \
    cin >> a[i];                                                               \
  }
#define F1(a, n)                                                               \
  for (int i = 1; i <= n; i++) {                                               \
    cin >> a[i];                                                               \
  }
#define P(a, n)                                                                \
  for (int i = 0; i < n; i++) {                                                \
    cout << a[i] << ' ';                                                       \
  }                                                                            \
  cout << endl;
#define P1(a, n)                                                               \
  for (int i = 1; i <= n; i++) {                                               \
    cout << a[i] << ' ';                                                       \
  }                                                                            \
  cout << endl;
#define NF(a, n, m)                                                            \
  for (int i = 0; i < n; i++) {                                                \
    for (int j = 0; j < m; j++) {                                              \
      cin >> a[i][j];                                                          \
    }                                                                          \
  }
#define NF1(a, n, m)                                                           \
  for (int i = 1; i <= n; i++) {                                               \
    for (int j = 1; j <= m; j++) {                                             \
      cin >> a[i][j];                                                          \
    }                                                                          \
  }
#define PNF(a, n, m)                                                           \
  for (int i = 0; i < n; i++) {                                                \
    for (int j = 0; j < m; j++) {                                              \
      cout << a[i][j] << ' ';                                                  \
    }                                                                          \
    cout << endl;                                                              \
  }                                                                            \
  cout << endl;
#define PNF1(a, n, m)                                                          \
  for (int i = 1; i <= n; i++) {                                               \
    for (int j = 1; j <= m; j++) {                                             \
      cout << a[i][j] << ' ';                                                  \
    }                                                                          \
    cout << endl;                                                              \
  }                                                                            \
  cout << endl;
#define AS 200001
#define mod 1000000007
int main() {
  fastIO string s;
  cin >> s;

  /**
   * Block Logic: Sorts the numbers in the sum.
   * The numbers are located at even indices of the string.
   * This code implements a bubble sort on the characters at these positions.
   */
  for (int i = 0; i < s.size(); i += 2)
    for (int j = 0; j < s.size(); j += 2)
      if (s[i] < s[j])
        swap(s[i], s[j]);
  cout << s;
}