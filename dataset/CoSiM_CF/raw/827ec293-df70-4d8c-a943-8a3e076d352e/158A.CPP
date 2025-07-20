#include <iostream>

using namespace std;

int main() {
  int m, k, x, s = 0;
  cin >> m >> k;
  int a[m];
  for (int i = 0; i < m; i++)
    cin >> a[i];
  for (int i = 0; i < m; i++)
    if (a[i] >= a[k - 1] && a[i] > 0)
      s++;
  cout << s;
  return 0;
}
