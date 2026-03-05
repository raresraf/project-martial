/**
 * @file 58A.CPP
 * @brief Solution for Codeforces Problem 58A, "Chat room".
 * @details This program checks if the word "hello" exists as a subsequence
 * within a given input string.
 *
 * Algorithm:
 * The code uses a greedy algorithm, which is an efficient method for solving
 * this type of subsequence problem.
 * 1. It maintains an index `inc` (initially 0) that points to the current
 *    character being sought in the target string "hello".
 * 2. It iterates through the input string `str` from left to right.
 * 3. If the current character from `str` matches the character at `hello[inc]`,
 *    it means the next character of the subsequence has been found, so `inc`
 *    is incremented.
 * 4. After scanning the entire input string, if `inc` has reached 5 (the
 *    length of "hello"), it signifies that all characters of the subsequence
 *    were found in the correct order.
 *
 * Time Complexity: O(N), where N is the length of the input string.
 * Space Complexity: O(N) for storing the input string.
 */
#include "bits/stdc++.h"

using namespace std;

int main(int argc, char const *argv[]) {
  string str;
  cin >> str;

  int inc = 0;
  string str2 = "hello";

  /**
   * @brief This loop iterates through the input string `str` to greedily
   * find the "hello" subsequence.
   */
  for (int i = 0; i < str.size(); i++) {
    /**
     * @brief If the current character of `str` matches the character we're
     * currently looking for in "hello", we advance our position (`inc`)
     * in the target string.
     */
    if (str[i] == str2[inc]) {
      inc++;
    }
  }

  /**
   * @brief If `inc` is 5, it means all five characters of "hello" were found
   * in the correct order.
   */
  if (inc == 5)
    cout << "YES";
  else
    std::cout << "NO";
  return 0;
}
