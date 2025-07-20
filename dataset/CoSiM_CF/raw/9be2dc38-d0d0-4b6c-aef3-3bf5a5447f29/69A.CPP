#include <bits/stdc++.h>

using namespace std;

int main() {

  int t;

  cin >> t;

  while (t--) {
    int n;
    cin >> n;

    int x[n + 2];

    for (int i = 0; i < n; i++) {
      cin >> x[i];
    }

    sort(x, x + n);

    int result = min(n - 2, x[n - 2] - 1);

    cout << result << endl;
  }
  return 0;
}
