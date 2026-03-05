/**
 * @file 58A.CPP
 * @brief Solution for Codeforces Problem 58A, "Chat room".
 * @details This program determines if the word "hello" can be formed as a
 * subsequence of a given input string.
 *
 * Algorithm:
 * The code uses a greedy approach, which is an efficient way to solve the
 * subsequence problem.
 * 1. It defines the target subsequence, "hello".
 * 2. It maintains a pointer or index `k` for the current character it is
 *    searching for in the target subsequence, starting with `k=0` ('h').
 * 3. It iterates through the input string `s` from left to right.
 * 4. If the current character in `s` matches the character `target[k]`, it
 *    means the next part of the subsequence has been found, so it increments `k`.
 * 5. After scanning the entire input string, if `k` is equal to the length
 *    of the target ("hello"), it means all characters were found in the
 *    correct order.
 *
 * Time Complexity: O(N), where N is the length of the input string.
 * Space Complexity: O(N) to store the input string.
 */
#include <bits/stdc++.h>
using namespace std;
int main() {
  string s, target;
  cin >> s;

  target = "hello";
  int n = s.length();

  // `k` is an index for the target string "hello".
  int k = 0;
  /**
   * @brief This loop iterates through the input string `s` to find the
   * "hello" subsequence greedily.
   */
  for (int i = 0; i < n; i++) {
    /**
     * @brief If the current character of `s` matches the character we are
     * looking for in "hello", we advance our target index `k`.
     */
    if (s[i] == target[k]) {
      k++;
    }
  }

  /**
   * @brief After the loop, if `k` equals the length of "hello", it means
   * we have successfully found all characters of the subsequence in order.
   */
  if (k == target.length()) {
    cout << "YES" << endl;
  } else {
    cout << "NO" << endl;
  }

  return 0;
}
