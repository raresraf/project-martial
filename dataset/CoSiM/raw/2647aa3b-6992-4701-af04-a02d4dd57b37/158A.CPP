#include <iostream>
using namespace std;

int main() {
  int n, k, i, ct = 0, a[50];
  cin >> n >> k;
  for (i = 0; i < n; ++i)
    cin >> a[i];
  for (i = 0; i < n; ++i)
    if ((a[i] >= a[k - 1]) && (a[i] > 0))
      ct++;
  cout << ct;
  return 0;
}
