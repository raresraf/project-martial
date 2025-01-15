#include <iostream>

using namespace std;

int main() {
  int n, k, i, a[50], w = 0;
  cin >> n >> k;
  for (i = 1; i <= n; i++)
    cin >> a[i];
  for (i = 1; i <= n; i++) {
    if (a[i] >= a[k] && a[i] > 0)
      w++;
  }
  cout << w << endl;

  return 0;
}
