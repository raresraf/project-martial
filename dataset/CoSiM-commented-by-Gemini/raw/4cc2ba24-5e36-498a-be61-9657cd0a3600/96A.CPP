/**
 * @file 96A.CPP
 * @brief Solves the "Football" problem from Codeforces.
 * @details The problem is to determine if a situation in a football match is
 * "dangerous". A situation is considered dangerous if there are at least seven
 * consecutive players from the same team. The teams are represented by '0' and '1'.
 *
 * @algorithm
 * This solution uses the `std::string::find` method, which is a highly
 * optimized and straightforward way to solve this problem.
 * 1. Read the entire sequence of players as a single string.
 * 2. Search for the literal substring "1111111".
 * 3. Search for the literal substring "0000000".
 * 4. If either substring is found, the condition is met, and the situation is
 *    dangerous. Otherwise, it is not.
 *
 * @complexity
 * Time Complexity: O(N), where N is the length of the input string. The
 * `std::string::find` method performs a linear scan of the string.
 * Space Complexity: O(N) to store the input string.
 */
#include <bits/stdc++.h>

using namespace std;

int main() {
  /**
   * @brief Optimizes C++ stream I/O.
   * `sync_with_stdio(false)` disables synchronization with C-style stdio.
   * `cin.tie(NULL)` unties `cin` from `cout`.
   * These are common performance optimizations in competitive programming.
   */
  ios_base::sync_with_stdio(false);
  cin.tie(NULL);

  /**
   * @brief Redirects I/O to files for local testing.
   * This block is compiled only when the `ONLINE_JUDGE` macro is not defined
   * (i.e., when running on a local machine instead of the judge's server).
   * It reads input from "input.txt", writes output to "output.txt", and
   * error streams to "error.txt".
   */
#ifndef ONLINE_JUDGE
  freopen("input.txt", "r", stdin);
  freopen("output.txt", "w", stdout);
  freopen("error.txt", "w", stderr);
#endif

  string str;
  cin >> str;

  /**
   * @block
   * @brief Core logic to detect a dangerous situation.
   * @details It checks for the existence of at least 7 consecutive '1's or '0's.
   * `string::npos` is a special value returned by `find` when the substring
   * is not found.
   */
  if ((str.find("1111111") != string::npos) ||
      (str.find("0000000") != string::npos))
    cout << "YES";
  else
    cout << "NO";

  /**
   * @brief Optional timing information for local debugging.
   * This line prints the execution time to the standard error stream, which
   * is useful for performance analysis during local development. It has no
   * effect on the solution's correctness on the judge.
   */
  cerr << "\ntime taken: " << (float)clock() / CLOCKS_PER_SEC << " secs" << endl;
  return 0;
}