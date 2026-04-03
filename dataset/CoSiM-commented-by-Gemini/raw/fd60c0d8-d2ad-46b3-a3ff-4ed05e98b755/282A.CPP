/**
 * @file 282A.cpp
 * @brief Solution to the "Bit++" competitive programming problem.
 * This program simulates a simple programming language with one variable, `X`,
 * initialized to 0, and two operations: `++` and `--`. It reads `n` statements
 * and calculates the final value of `X`.
 *
 * The file starts with several macros and typedefs, which are common shortcuts
 * in competitive programming for faster coding.
 *
 * Algorithm:
 * 1. Initialize a variable `ans` (representing X) to 0.
 * 2. Read the number of statements `n`.
 * 3. Loop `n` times:
 *    a. Read the statement as a string.
 *    b. Check the second character of the string (`tmp[1]`).
 *       - If it is '+', increment `ans`.
 *       - If it is '-', decrement `ans`.
 *       This works because for both "++X" and "X++", the '+' is at index 1.
 *       The same logic applies to "--X" and "X--".
 * 4. Print the final value of `ans`.
 *
 * Time Complexity: O(N), where N is the number of statements.
 * Space Complexity: O(1) (excluding storage for the input strings, which are processed one by one).
 */
#pragma comment(linker, "/STACK:16777216")
#include <bits/stdc++.h>

using namespace std;

// Common macros for competitive programming.
#define ms(ar, a) memset(ar, a, sizeof(ar))
#define fr(i, j, k) for (int(i) = (j); (i) < (k); (i)++)
#define db(x) cout << (#x) << " = " << x << endl;
#define pb push_back
#define mp make_pair
#define X first
#define Y second

// Typedefs for common data types.
typedef long long ll;
typedef pair<int, int> pii;

template <class _T> inline string tostr(const _T &a) {
  ostringstream os("");
  os << a;
  return os.str();
}

int n;

int main() {

  scanf("%d", &n);
  int ans = 0; // This variable will hold the final value of X.

  // Loop through each statement.
  while (n--) {
    string tmp;
    cin >> tmp;
    // Check the second character to determine the operation.
    if (tmp[1] == '+')
      ans++;
    else
      ans--;
  }

  printf("%d\n", ans);

  return 0;
}
