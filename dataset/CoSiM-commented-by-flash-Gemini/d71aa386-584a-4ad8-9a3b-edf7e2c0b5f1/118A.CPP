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

  string t;

  int i = 0;
  cin >> t;

  while (i < t.length()) {

    if (!(t[i] == 'a' || t[i] == 'e' || t[i] == 'i' || t[i] == 'o' ||
          t[i] == 'u' || t[i] == 'y'))
      t[i] = tolower(t[i]);

    if (!(t[i] == 'a' || t[i] == 'e' || t[i] == 'i' || t[i] == 'o' ||
          t[i] == 'u' || t[i] == 'y')) {
      cout << "." << t[i];
    }
    i++;
  }
}
