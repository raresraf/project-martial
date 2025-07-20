#include <iostream>
using namespace std;

int main() {
  int i, n, k, a[50], s = 0;
  cin >> n >> k;
  for (i = 0; i < n; i++)
    cin >> a[i];
  for (i = 0; i < n; i++) {
    if (a[i] >= a[k - 1] && a[i] > 0)
      s++;
  }
  cout << s;
  return 0;
}
