#include <iostream>
#include <string.h>

using namespace std;

int main() {
  int n, t;

  cin >> n >> t;

  char a[n];

  cin >> a;

  while (t > 0) {
    for (int i = 1; i < n; ++i) {
      if (a[i] == 'G' && a[i - 1] == 'B') {
        char aux;
        aux = a[i - 1];
        a[i - 1] = a[i];
        a[i] = aux;
        i++;
      }
    }
    t--;
  }

  cout << a << endl;
}