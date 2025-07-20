#include <algorithm>
#include <iostream>
using namespace std;

int main() {
  int n, k;
  cin >> n >> k;

  int a[n];

  for (int i = 0; i < n; i++)
    cin >> a[i];

  int i;
  if (a[k - 1] > 0) {
    for (i = k; i < n && a[i] == a[k - 1]; i++)
      ;
    cout << i << endl;
  } else {
    for (i = k - 1; i >= 0 && a[i] == 0; i--)
      ;
    if (i < 0)
      cout << "0\n";
    else
      cout << i + 1 << endl;
  }

  return 0;
}
