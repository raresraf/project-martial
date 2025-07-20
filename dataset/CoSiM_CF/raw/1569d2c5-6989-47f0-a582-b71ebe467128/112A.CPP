// Codeforces_Andrespengineer
/* List of solved problems
 *
 * Andr s Sierra
 * Instituto Tecnol gico de Santo Domingo (INTEC)
 *
 */

#include <cstdlib>
#include <iostream>
#include <string>

using namespace std;

int main(int argc, char *argv[]) {
  string s, t;
  cin >> s;
  cin >> t;
  int i = 0;

  while (i < s.length()) {
    s[i] = tolower(s[i]);
    t[i] = tolower(t[i]);

    if (s[i] == t[i])
      i++;

    else {
      if (s[i] > t[i]) {
        cout << 1;
      }

      if (t[i] > s[i]) {
        cout << -1;
      }

      break;
    }

    if (i == s.length())
      cout << 0;
  }
}
