/**
 * @file 96A.CPP
 * @brief Solves the "Football" problem from Codeforces.
 *
 * This program determines if a given situation is "dangerous" by checking if
 * there are at least 7 consecutive players of the same team. The team players
 * are represented by '0's and '1's in a string.
 *
 * @algorithm The code implements a linear scan through the input string. It uses a
 * counter to track the number of consecutive identical characters. If a different
 * character is encountered, the counter is reset. If the counter reaches 7 at any
 * point, the condition is met, and the program can conclude.
 *
 * @time_complexity O(N), where N is the length of the input string, as it involves a single pass.
 * @space_complexity O(N) to store the input string.
 */
#include <bits/stdc++.h>

using namespace std;
// Unused macro definitions for general competitive programming.
#define EPS (1e-7)
#define INF (1e9)
#define PI (acos(-1))

typedef long long ll;
#define endl '\n'

// Unused array definitions, likely for grid-based problems.
int dx[4] = {1, 0, -1, 0}, dy[4] = {0, 1, 0, -1};

const int MOD = 1000000007;

//////////////////////////////////////////

string s;
/**
 * @brief Reads the input string from standard input.
 */
void Input() { cin >> s; }

/**
 * @brief Scans the string to find if there are 7 or more consecutive identical characters.
 */
void Solve() {

  char c = s[0];
  int cnt = 1;
  int len = s.size();
  /**
   * @block_logic Iterates through the string to count consecutive characters.
   */
  for (int i = 1; i < len; i++) {

    // Optimization: If a dangerous situation is already found, exit the loop early.
    if (cnt >= 7) {

      break;
    }
    // If the current character is the same as the previous one, increment the counter.
    if (c == s[i]) {

      cnt++;

    } else {

      // If a different character is found, reset the counter and update the character being tracked.
      c = s[i];
      cnt = 1;
    }
  }

  // After the loop, check if the count ever reached 7 or more.
  if (cnt >= 7)
    cout << "YES" << endl;
  else
    cout << "NO" << endl;
}

int main() {

  // Optimizes I/O operations.
  cin.tie(0);
  ios::sync_with_stdio(false);
  Input();
  Solve();

  return 0;
}
