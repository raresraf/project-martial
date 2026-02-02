#include <iostream>

using namespace std;
/**
 * @brief This program determines if a football game situation is "dangerous".
 *
 * A situation is considered dangerous if there are at least 7 consecutive players
 * from the same team. The team alignment is represented by a binary string, where
 * '0' and '1' represent players from the two different teams.
 *
 * The program reads a binary string from the standard input and checks for a
 * sequence of 7 identical characters. If such a sequence is found, it prints
 * "YES" and terminates. Otherwise, it prints "NO" after checking the entire
 * string.
 */
int main() {
  int count = 1;
  string s;
  cin >> s;

  for (int i = 1; i < s.size(); i++) {
    if (s[i] == s[i - 1]) {
      count++;
      if (count == 7) {
        cout << "YES";
        return 0;
      }
    } else
      count = 1;
  }
  cout << "NO";
  return 0;
}