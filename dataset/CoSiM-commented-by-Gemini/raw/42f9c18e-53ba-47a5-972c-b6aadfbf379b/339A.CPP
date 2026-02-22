/**
 * @file 339A.CPP
 * @brief Solution to Codeforces problem 339A - Helpful Maths.
 *
 * @algorithm
 * The problem requires sorting the numbers (1, 2, or 3) in a string that
 * represents a sum, like "3+1+2", into non-decreasing order, like "1+2+3".
 *
 * This solution uses a standard and robust "extract, sort, reprint" algorithm:
 * 1. Read the input string.
 * 2. Create a `std::vector<int>` to store the numbers.
 * 3. Iterate through the input string, ignoring the '+' characters and converting
 *    the digit characters to integers. Add these integers to the vector.
 * 4. Use `std::sort` to sort the vector of numbers.
 * 5. Iterate through the sorted vector and print the elements, separated by '+',
 *    ensuring no '+' is printed after the last element.
 *
 * @note The code is wrapped in a `while (cin >> str)` loop, which is a common
 *       pattern for handling multiple test cases, although this problem only
 *       has one. It also features many custom preprocessor macros for brevity.
 *
 * @complexity
 * - Time Complexity: O(L + N log N), where L is the string length and N is the
 *   number of summands. O(L) for parsing and O(N log N) for sorting.
 * - Space Complexity: O(N), to store the numbers in the vector.
 */

/***

            Bismillahir Rahmanir Rahim
            Read the name of Allah, who created you!!!
            Author : Shah Newaj Rabbi Shishir
            Department of CSE, City University, Bangladesh.

***/

#include <bits/stdc++.h>
using namespace std;

#define sf scanf
#define pf printf
#define scase sf("%d", &tc)
#define sn sf("%d", &n)
#define whilecase while (tc--)
#define eof while (cin >> n)
#define forloop for (pos = 1; pos <= tc; pos++)
#define arrayloop (i = 0; i < n; i++)
#define cinstr cin >> str
#define getstr getline(cin, str)
#define pcase pf("Case %d: ", pos)
#define pb push_back
#define in insert
#define llu unsigned long long
#define lld long long
#define u unsigned int

int main(void) {
  string str;
  int len, i, val, siz;

  // This loop will run once for the given problem constraints.
  while (cin >> str) {
    len = str.length();

    vector<int> v;

    /**
     * @brief Stage 1: Extract numbers from the string.
     * It iterates through the string, skipping '+' and converting digits to ints.
     */
    for (i = 0; i < len; i++) {
      if (str[i] == '+')
        continue;

      val = str[i] - '0';
      v.pb(val);
    }

    /**
     * @brief Stage 2: Sort the extracted numbers.
     */
    sort(v.begin(), v.end());

    siz = v.size();

    /**
     * @brief Stage 3: Print the sorted numbers to reconstruct the sum.
     */
    for (i = 0; i < siz; i++) {
      if (i == siz - 1)
        cout << v[i] << endl; // Last element, no trailing '+'.
      else
        cout << v[i] << "+";
    }
  }

  return 0;
}
