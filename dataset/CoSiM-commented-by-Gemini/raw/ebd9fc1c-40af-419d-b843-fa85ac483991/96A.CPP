/**
 * @file 96A.CPP
 * @brief Solves the "Football" problem from Codeforces.
 *
 * This program checks if a string of '0's and '1's, representing players,
 * contains a "dangerous" situation, defined as 7 or more consecutive
 * players of the same team.
 *
 * @algorithm The code performs a linear scan of the input string. It uses a
 * state variable `curr` to keep track of the character in the current run and
 * a `count` for the length of that run. When the character changes, the state is
 * reset. If the count reaches 7 at any point, a flag is set, and the loop
 * terminates.
 *
 * @time_complexity O(N), where N is the length of the input string.
 * @space_complexity O(N) to store the input string.
 */
#include <bits/stdc++.h>

using namespace std;

int main() {
  string s;
  cin >> s;
  // `check` flag is true if the situation is NOT dangerous.
  bool check = true;
  char curr = s[0];
  int count = 1;
  /**
   * @block_logic Iterates through the string to find a run of 7 identical characters.
   */
  for (int i = 1; i < s.length(); i++) {
    if (s[i] == curr) {
      // If the character is the same as the previous one, continue the run.
      count++;
    } else {
      // If the character changes, start a new run.
      curr = s[i];
      count = 1;
    }
    // If a run of 7 is found, the situation is dangerous.
    if (count == 7) {
      check = false;
      break;
    }
  }
  // Print the result based on the final state of the flag.
  if (check)
    cout << "NO";
  else
    cout << "YES";
}