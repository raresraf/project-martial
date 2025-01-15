#include <algorithm>
#include <iostream>

using namespace std;

int main() {
  int n;
  cin >> n;
  int a[n], s = 0;
  for (int i = 0; i < n; i++) {
    cin >> a[i];
    s += a[i];
  }
  sort(a, a + n);
  int c = 0, f = 0;
  for (int i = n - 1; i >= 0; i--) {
    c += a[i];
    f++;
    if (c > (s - c)) {
      cout << f << endl;
      break;
    }
  }
  return 0;
}
