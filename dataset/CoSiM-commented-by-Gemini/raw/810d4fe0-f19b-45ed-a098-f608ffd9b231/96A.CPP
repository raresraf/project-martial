// 4007191   Jul 3, 2013 7:32:32 PM	fuwutu	 96A - Football	 GNU C++0x
// Accepted	15 ms	0 KB
#include <iostream>
#include <string>

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
  string s;
  cin >> s;

  int contiguous = 1;
  for (size_t i = 1; i < s.length(); ++i) {
    if (s[i] == s[i - 1]) {
      contiguous += 1;
      if (contiguous == 7) {
        cout << "YES" << endl;
        return 0;
      }
    } else {
      contiguous = 1;
    }
  }

  cout << "NO" << endl;
  return 0;
}