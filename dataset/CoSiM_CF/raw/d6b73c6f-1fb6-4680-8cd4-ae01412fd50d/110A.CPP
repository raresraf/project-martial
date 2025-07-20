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

  long long n;

  cin >> n;
  int LuckyDigits = 0;

  while (n > 0) {

    if (n % 10 == 7 || n % 10 == 4) {
      LuckyDigits++;
    }

    n = n / 10;
  }

  if (LuckyDigits == 4 || LuckyDigits == 7)
    cout << "YES" << endl;
  else
    cout << "NO" << endl;
}
