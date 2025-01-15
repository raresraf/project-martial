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
  string s;

  cin >> s;

  int flag = 0;
  int l = s.length();
  int n = 0;

  for (int i = 0; i < l; i++) {
    n = 1;

    for (int j = i + 1; j < l; j++) {

      if (s[i] == s[j])
        n++;
      else
        break;
    }

    if (n >= 7) {

      cout << "YES";
      flag = 1;
      break;
    }
  }
  if (flag == 0)
    cout << "NO";
}
