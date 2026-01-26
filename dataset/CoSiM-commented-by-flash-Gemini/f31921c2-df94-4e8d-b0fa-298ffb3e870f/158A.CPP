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

int main() {

  int n, k, p, c = 0;

  cin >> n >> k;

  int arr[n];

  for (int i = 0; i < n; i++) {

    cin >> p;
    arr[i] = p;
  }

  k = k - 1;

  while (k + 1 < n) {
    if (arr[k] == arr[k + 1])
      k++;
    else
      break;
  }
  while (k >= 0) {
    if (arr[k] == 0)
      k--;
    else
      break;
  }

  cout << k + 1 << endl;
}
