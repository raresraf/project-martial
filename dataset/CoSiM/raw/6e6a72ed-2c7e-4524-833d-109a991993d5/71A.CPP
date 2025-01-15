// Codeforces_Andrespengineer
/* List of solved problems
 *
 * Andr s Sierra
 * Instituto Tecnol gico de Santo Domingo (INTEC)
 *
 */

#include <cstdlib>
#include <iostream>

using namespace std;

int main(int argc, char *argv[]) {
  string s;
  int n;

  cin >> n;

  if (n >= 1 && n <= 100) {
    while (n > 0) {
      cin >> s;

      if (s.length() > 10)
        cout << s[0] << s.length() - 2 << s[s.length() - 1] << "\n";
      else
        cout << s << "\n";
      n--;
    }
  }
}
