// Codeforces_Andrespengineer
/* List of solved problems
 *
 * Andr s Sierra
 * Instituto Tecnol gico de Santo Domingo (INTEC)
 *
 */

#include <algorithm>
#include <cstdlib>
#include <iostream>
#include <vector>

using namespace std;

long sumar(vector<int> vec, int cnt) {
  int suma = 0;

  vector<int>::iterator it = vec.begin() + cnt;
  for (; it < vec.end(); it++) {
    suma = suma + *it;
  }

  return (suma);
}

vector<int> vec;

int main() {

  int n;
  int value;
  long sum = 0;
  int cnt = 0;
  int min = 0;
  cin >> n;

  while (cin >> value) {
    vec.push_back(value);
    n--;
  }

  sort(vec.begin(), vec.end());
  reverse(vec.begin(), vec.end());

  vector<int>::iterator it = vec.begin();

  for (; it < vec.end(); it++) {
    sum = sum + *it;

    cnt++;

    if (sum <= sumar(vec, cnt))
      min++;

    else {

      min++;
      cout << min;
      break;
    }
  }

  cout << endl;
}
