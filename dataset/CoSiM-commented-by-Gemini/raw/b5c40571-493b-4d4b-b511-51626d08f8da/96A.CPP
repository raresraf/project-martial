/**
 * @file 96A.CPP
 * @brief Solution to the "Football" problem.
 * @see https://codeforces.com/problemset/problem/96/A
 *
 * This program determines if a football team has a "dangerous" situation,
 * which is defined as having at least 7 consecutive players of the same team.
 * The teams are represented by '0's and '1's in a string.
 *
 * Algorithm:
 * 1. Read the string of players.
 * 2. Iterate through the string, keeping a count of consecutive identical
 *    characters.
 * 3. If the count reaches 7, the situation is dangerous, and the program
 *    prints "YES" and terminates.
 * 4. If a different character is encountered, the counter is reset to 1.
 * 5. If the end of the string is reached without the count reaching 7, the
 *    situation is not dangerous, and the program prints "NO".
 *
 * Time Complexity: O(N), where N is the length of the string.
 * Space Complexity: O(N) to store the input string.
 */
#include <bits/stdc++.h>
using namespace std;
int main() {
  int count = 1;
  string str;
  cin >> str;
  // Invariant: Iterates through the string to find a sequence of 7 identical
  // characters.
  for (int i = 1; i < str.length(); i++) {
    // Pre-condition: If the current character is the same as the previous one,
    // increment the counter.
    if (str[i] == str[i - 1]) {
      count += 1;
      // Pre-condition: If 7 consecutive identical characters are found, the
      // condition is met.
      if (count == 7) {
        cout << "YES" << endl;
        return 0;
      }
    } else {
      count = 1;
    }
  }
  cout << "NO" << endl;
}