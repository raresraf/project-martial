#include <iostream>
using namespace std;
int main() {
  int n, k, i, j;
  int a[101];
  while (cin >> n >> k) {
    for (i = 0; i < n; i++) {
      cin >> a[i];
    }
    j = 0;
    for (i = 0; i < n; i++) {
      if ((a[i] >= a[k - 1]) && a[i] > 0) {
        j++;
      }
    }
    cout << j << endl;
  }
  return 0;
}
