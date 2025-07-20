// Codeforces_Andrespengineer
/* List of solved problems
 *
 * Andr s Sierra
 * Instituto Tecnol gico de Santo Domingo (INTEC)
 *
 */

#include <iostream>

using namespace std;

int main() {

  int n, stones = 0;
  int i = 0;

  cin >> n;

  int t = n;
  char c[n];

  while (n > 0) {
    cin >> c[i];
    i++;
    n--;
  }

  i = 0;
  n = t;

  while (n > 0) {
    if (c[i] == c[i + 1])
      stones++;

    i++;
    n--;
  }

  cout << stones << endl;
}
