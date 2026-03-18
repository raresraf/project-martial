/**
 * @file 96A.CPP
 * @brief This program determines if a given string contains a "dangerous" situation,
 * defined as having 7 or more consecutive identical characters.
 * Algorithm: Single-pass iterative scan with a counter for consecutive identical characters.
 * Time Complexity: O(L) where L is the length of the input string.
 * Space Complexity: O(1) as only a few variables are used to store state.
 */
#include <bits/stdc++.h>

using namespace std;

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

/**
 * @brief Main function to execute the dangerous situation detection.
 * It reads an input string and checks for a sequence of 7 or more identical characters.
 */
int main() {
  string s;
  cin >> s;
  char curr = '0';
  int currl = 0;
  /**
   * @brief Iterates through the input string to count consecutive identical characters.
   * Pre-condition: 's' is the input string. 'curr' stores the character being tracked,
   * 'currl' counts its consecutive occurrences.
   * Invariant: At each iteration, 'currl' reflects the current count of consecutive
   * characters ending at the current index.
   */
  for (int i = 0; i < (int)s.length(); i++) {
    /**
     * @brief Updates the count of consecutive identical characters.
     * Logic: If the current character matches the previously tracked character, increments 'currl'.
     * Otherwise, resets 'currl' to 1 for the new character.
     */
    if (curr == s[i])
      currl++;
    else
      currl = 1;
    curr = s[i];
    /**
     * @brief Checks if a dangerous situation (7 or more consecutive identical characters) is detected.
     * Pre-condition: 'currl' holds the current count of consecutive identical characters.
     * Post-condition: If 'currl' is 7 or more, prints "YES" and terminates the program.
     */
    if (currl >= 7) {
      cout << "YES" << endl;
      return 0;
    }
  }
  cout << "NO" << endl;
  return 0;
}
