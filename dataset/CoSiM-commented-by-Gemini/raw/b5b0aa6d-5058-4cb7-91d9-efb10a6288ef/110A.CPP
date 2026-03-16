/**
 * @file 110A.CPP
 * @brief Solution to the "Nearly Lucky Number" problem (Codeforces 110A).
 * @author samikshamodi
 *
 * @details
 * This program determines if a given number is "nearly lucky". A number is considered
 * nearly lucky if the count of its "lucky digits" ('4' and '7') is itself a lucky number.
 * A lucky number is a positive integer whose decimal representation contains only the digits 4 and 7.
 *
 * The program first reads the input number as a string. It then iterates through the
 * string to count the total number of '4's and '7's. Finally, it checks if this count
 * is a lucky number.
 *
 * Given the problem constraints (the input number has between 1 and 18 digits), the maximum
 * possible count of lucky digits is 18. The only lucky numbers within this range are 4 and 7.
 * Therefore, the program only needs to check if the count is equal to 4 or 7.
 *
 * Algorithm:
 * 1. Read the number as a string.
 * 2. Count the occurrences of '4' and '7'.
 * 3. Check if the count is 4 or 7.
 *
 * Time Complexity: O(L), where L is the number of digits in the input number.
 * Space Complexity: O(L) to store the input string.
 */
/* samikshamodi - Samiksha Modi */
#include <bits/stdc++.h>
using namespace std;
typedef long long ll;
typedef pair<int, int> pii;

#define endl '\n'
#define pb push_back
#define mp make_pair
#define f first
#define s second

#define sync                                                                   \
  ios_base::sync_with_stdio(false);                                            \
  cin.tie(NULL);
#define out1d(arr, size)                                                       \
  {                                                                            \
    for (int i = 0; i < size; i++)                                             \
      cout << arr[i] << " ";                                                   \
  }
#define in2d(arr, row, col)                                                    \
  for (int i = 0; i < row; i++)                                                \
    for (int j = 0; j < col; j++)                                              \
      cin >> arr[i][j];
#define out2d(arr, row, col)                                                   \
  {                                                                            \
    for (int i = 0; i < row; i++) {                                            \
      for (int j = 0; j < col; j++)                                            \
        cout << arr[i][j] << " ";                                              \
      cout << endl;                                                            \
    }                                                                          \
  }
#define mem(arr, val) memset(arr, val, sizeof(arr));

const ll MOD = 1e9 + 7;
const ll INF = 1e18;
const ll N = 1e5 + 7;

int main() {
  sync;
  string s;
  cin >> s;
  int lcnt = 0;

  /**
   * @brief Counts the number of lucky digits ('4' or '7') in the input string.
   */
  for (int i = 0; i < s.length(); i += 1) {
    if (s[i] == '4' || s[i] == '7')
      lcnt++;
  }

  /**
   * @brief Determines if the count of lucky digits is itself a lucky number.
   * @details Based on the problem constraints, the count can be at most 18.
   * The only lucky numbers <= 18 are 4 and 7. Therefore, we only need to
   * check for these two values.
   */
  if (lcnt == 4 || lcnt == 7)
    cout << "YES";
  else
    cout << "NO";

  return 0;
}
