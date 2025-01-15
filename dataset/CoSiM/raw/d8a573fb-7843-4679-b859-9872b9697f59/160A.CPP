#include <algorithm>
#include <iostream>

using namespace std;

int main() {
  int n, a[105], i = 0, k = 0, m = 0;

  cin >> n;

  for (i = 0; i < n; i++) {
    cin >> a[i];

    m += a[i];
  }

  sort(a, a + n);

  for (; i--;) {
    k += a[i];

    if (k * 2 > m)
      break;
  }

  cout << n - i << endl;

  return 0;
}
