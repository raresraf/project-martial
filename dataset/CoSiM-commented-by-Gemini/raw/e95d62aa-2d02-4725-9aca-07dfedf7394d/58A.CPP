/**
 * @file 58A.CPP
 * @brief Solves the "Chat room" problem from Codeforces.
 *
 * This program determines if the word "hello" can be formed by deleting zero or
 * more characters from a given input string. In other words, it checks if
 * "hello" is a subsequence of the input string.
 *
 * @algorithm The code uses a greedy, two-pointer approach. A pointer `x` tracks
 * the current character to find in the target string "hello". The main loop
 * iterates through the input string. When a character matching the target
 * character is found, the target pointer `x` is advanced. If `x` successfully
 * traverses all of "hello", the subsequence exists.
 *
 * @time_complexity O(N), where N is the length of the input string.
 * @space_complexity O(1), constant extra space is used.
 */
#include <iostream>
using namespace std;

int main() {
  string word;
  cin >> word;
  int n = word.length();
  string goal = "hello";
  int x = 0; // Pointer for the 'goal' string.
  string result = "NO";
  /**
   * @block_logic Iterates through the input string `word` to find the 'goal'
   * string as a subsequence.
   */
  for (int i = 0; i < n; i++) {
    // If the current character in `word` matches the current character in `goal`...
    if (word[i] == goal[x]) {
      x++; // ...advance the pointer for the goal string.
    }
    // If we have found all characters of "hello" in order.
    if (x == 5) {
      result = "YES";
    }
  }
  cout << result;
  cout << "\n";
}
