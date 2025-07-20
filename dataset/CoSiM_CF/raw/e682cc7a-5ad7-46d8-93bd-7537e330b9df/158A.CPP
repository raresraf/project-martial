#include <iostream>

using namespace std;

int main() {
  int n, k, a[100];
  int w(0);

  cin >> n >> k;

  if (!(1 <= k && k <= n && n <= 50)) {
    return 0;
  }

  for (int i = 1; i <= n; i++) {
    cin >> a[i];

    if (!(0 <= a[i] && a[i] <= 100) || (i != 1 && a[i] > a[i - 1])) {
      return 0;
    }
  }
  for (int i = 1; i <= n; i++) {
    if (a[i] > 0 && a[i] >= a[k]) {
      w++;
    }
  }
  cout << w << endl;
  return 0;
}