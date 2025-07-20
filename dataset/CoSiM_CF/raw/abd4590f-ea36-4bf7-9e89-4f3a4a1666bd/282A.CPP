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
  int n, value = 0;

  cin >> n;

  string temp;

  while (n > 0) {
    cin >> temp;
    if (temp[1] == '+') {
      value++;
    } else {
      value--;
    }
    n--;
  }

  cout << value << endl;
}
