/**
 * @file 96A.CPP
 * @brief Solution for Codeforces Problem 96A - "Football".
 * @details The program determines if a situation is "dangerous" in a football match.
 * A situation is considered dangerous if there are at least 7 consecutive players
 * from the same team. The team lineup is given as a string of '0's and '1's.
 *
 * @author (unknown)
 *
 * @algorithm
 * The program uses a single pass to find a consecutive run of characters.
 * 1. It reads the input string `s`.
 * 2. It maintains two variables: `curr` to store the character of the current run
 *    (e.g., '0' or '1'), and `currl` to store the length of that run.
 * 3. It iterates through the string character by character.
 * 4. If the current character is the same as `curr`, it increments `currl`.
 * 5. If the character is different, it signals the start of a new run, so `currl` is reset to 1.
 * 6. `curr` is updated to the current character.
 * 7. After each character is processed, it checks if `currl` has reached 7. If it has,
 *    a dangerous situation is found, "YES" is printed, and the program terminates.
 * 8. If the loop completes without `currl` ever reaching 7, it means no such run exists,
 *    and "NO" is printed.
 *
 * @note The file includes a template with overloaded `<<` operators for easy printing
 * of standard library containers, a common feature in competitive programming, though
 * they are not used in this particular solution.
 *
 * @complexity
 * Time Complexity: O(N), where N is the length of the string, due to the single pass.
 * Space Complexity: O(N) to store the input string.
 */
#include <bits/stdc++.h>

using namespace std;

// The following are template definitions for stream operators, often used in
// competitive programming for debugging purposes to easily print containers.
typedef long long ll;
typedef vector<int> vi;
typedef pair<int, int> pii;
template <typename T1, typename T2>
ostream &operator<<(ostream &os, pair<T1, T2> p) {
  os << p.first << " " << p.second;
  return os;
}
template <typename T> ostream &operator<<(ostream &os, vector<T> &v) {
  for (T i : v)
    os << i << ", ";
  return os;
}
template <typename T> ostream &operator<<(ostream &os, set<T> s) {
  for (T i : s)
    os << i << ", ";
  return os;
}
template <typename T1, typename T2>
ostream &operator<<(ostream &os, map<T1, T2> m) {
  for (pair<T1, T2> i : m)
    os << i << endl;
  return os;
}

int main() {
  string s;
  cin >> s;
  char curr = '0';
  int currl = 0;
  /**
   * Block Logic: This loop iterates through the string to find a run of at least 7
   * identical consecutive characters.
   */
  for (int i = 0; i < (int)s.length(); i++) {
    if (curr == s[i]) {
      // If the character is the same as the previous one, extend the current run.
      currl++;
    } else {
      // If the character changes, start a new run.
      currl = 1;
    }
    // Update the character for the current run.
    curr = s[i];
    // Check if a dangerous situation has occurred.
    if (currl >= 7) {
      cout << "YES" << endl;
      return 0; // Terminate early as the condition is met.
    }
  }
  // If the loop completes, no run of 7 was found.
  cout << "NO" << endl;
  return 0;
}
