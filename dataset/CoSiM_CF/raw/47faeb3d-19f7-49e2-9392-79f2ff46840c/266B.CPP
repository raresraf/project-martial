#include <iostream>
using namespace std;
int main() {
  int n, t, i, j;
  cin >> n;
  cin >> t;
  string a;
  cin >> a;
  for (i = 1; i <= t; i++) {
    for (j = 0; j < (n - 1); j++) {
      if (a[j] == 'B' && a[j + 1] == 'G') {
        a[j + 1] = 'B';
        a[j] = 'G';
        j++;
      }
    }
  }
  cout << a;
  return 0;
}