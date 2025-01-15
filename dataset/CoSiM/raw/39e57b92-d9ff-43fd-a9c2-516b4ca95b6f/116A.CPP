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
  int n, capacity = 0;
  int a, b, temp;

  cin >> n;

  int flag = n;

  while (n > 0) {
    cin >> a >> b;

    if (n == flag) {
      capacity = b;
      temp = b;
    }

    else {
      temp = (temp - a) + b;

      if (temp >= capacity)
        capacity = temp;
    }

    n--;
  }

  cout << capacity << endl;
}
