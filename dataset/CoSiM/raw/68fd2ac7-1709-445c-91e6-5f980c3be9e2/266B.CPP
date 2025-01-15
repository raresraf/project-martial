#include <iostream>
using namespace std;
int main() {
  int n, t;
  cin >> n >> t;
  int i, j, k;
  char a[50];
  cin >> a;
  for (i = 0; i < t; i++) {
    for (j = 0; j < n - 1; j++) {
      if (a[j] == 'B' && a[j + 1] == 'G') {
        a[j] = 'G';
        a[j + 1] = 'B';
        j++;
      }
    }
  }
  cout << a << endl;
  return 0;
}
