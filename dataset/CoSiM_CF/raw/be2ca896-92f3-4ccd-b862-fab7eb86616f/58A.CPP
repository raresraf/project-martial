// Codeforces_Andrespengineer
/* List of solved problems
 *
 * Andr s Sierra
 * Instituto Tecnol gico de Santo Domingo (INTEC)
 *
 */

#include <iostream>
#include <string>

using namespace std;

int main() {

  string str;

  cin >> str;

  int a = 0, b = 0, c = 0, d = 0, e = 0;

  a = str.find('h');
  b = str.find('e', a);
  c = str.find('l', b);
  d = str.find('l', c + 1);
  e = str.find('o', d);

  if (e > d && d > c && c > b && b > a) {
    cout << "YES" << endl;
  }

  else
    cout << "NO" << endl;
}
