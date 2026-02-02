#include <bits/stdc++.h>

using namespace std;

#define deb(x) cout << #x << " = " << x << '\n';
#define ll long long
#define PI 3.1415926535897932384626

void crap();

/**
 * @brief Solves the "Stones on the Table" problem.
 *
 * This program calculates the minimum number of stones to remove from a row
 * of colored stones such that no two adjacent stones have the same color.
 * The colors are represented by a string of characters.
 *
 * It reads the number of stones and a string representing their colors.
 * Then, it iterates through the string, counting the number of adjacent
 * stones with the same color. This count represents the number of stones
 * that need to be removed.
 *
 * @return 0 on successful execution.
 */
int main() {
  crap();

  int n, count;
  string s;
  cin >> n >> s;
  count = 0;
  for (int i = 1; i < n; ++i) {
    if (s[i] == s[i - 1]) {
      ++count;
    }
  }
  cout << count;

  return 0;
}

/**
 * @brief A utility function to speed up I/O operations.
 *
 * This function is commonly used in competitive programming to disable
 * synchronization with C-style I/O and untie cin and cout from the
 * standard I/O streams, which can significantly improve the performance
 * of input and output operations.
 */
void crap() { ios_base::sync_with_stdio(false), cin.tie(0), cout.tie(0); }
