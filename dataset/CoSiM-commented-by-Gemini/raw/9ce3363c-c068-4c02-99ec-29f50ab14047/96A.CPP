/**
 * @file 96A.CPP
 * @brief Determines if a string contains a sequence of 7 or more identical characters.
 *
 * This program solves a problem often related to game situations, like determining if
 * a player has a "dangerous" position. It reads a string of characters (representing,
 * for example, players in a line) and checks if there is a contiguous block of
 * 7 or more identical characters.
 *
 * @algorithm The core of the program is the function `f`, which finds the length
 * of the longest run of identical consecutive characters in a string. It iterates
 * through the string, keeping track of the current run's length (`cnt`) and the
 * maximum run found so far (`mx`). The main logic then checks if this maximum
 * run length is at least 7.
 *
 * @time_complexity O(N), where N is the length of the string, because the `f`
 * function iterates through the string once.
 * @space_complexity O(N) to store the input string.
 */
#include <cmath>
#include <iostream>
#include <string>

using namespace std;

/**
 * @brief Calculates the length of the longest run of identical consecutive characters.
 * @param s The input string (passed by reference).
 * @return The length of the longest run.
 * @note The loop condition `i <= s.length()` will cause an out-of-bounds access
 * on the final iteration when `i` equals `s.length()`. In C++, string indices are
 * from 0 to `length() - 1`. Accessing `s[s.length()]` is undefined behavior.
 * For typical competitive programming test cases, this might not cause a crash, but it is incorrect.
 */
int f(string &s) {
  int mx = 1, cnt = 1; // mx: max run length, cnt: current run length
  
  /**
   * @brief Loop to iterate through the string and find the longest run.
   * @invariant `mx` holds the maximum run length found in the substring `s[0...i-1]`.
   * `cnt` holds the run length ending at `s[i-1]`.
   */
  for (int i = 1; i <= s.length(); i++) {
    // Compare the current character with the previous one.
    if (s[i] == s[i - 1])
      cnt++; // If they are the same, increment the current run counter.
    else
      cnt = 1; // Otherwise, reset the counter to 1.
    
    // Update the maximum run length found so far.
    if (cnt > mx)
      mx = cnt;
  }
  return mx;
}

/**
 * @brief Main logic function to read input and print the result.
 */
void solve() {
  string s;
  cin >> s;
  
  /**
   * @brief Check if the team situation is "dangerous".
   * @pre The function `f(s)` returns the length of the longest run of identical characters.
   */
  if (f(s) >= 7)
    cout << "YES\n"; // The situation is dangerous.
  else
    cout << "NO\n";  // The situation is not dangerous.
}

int main() {
  solve();
  return 0;
}