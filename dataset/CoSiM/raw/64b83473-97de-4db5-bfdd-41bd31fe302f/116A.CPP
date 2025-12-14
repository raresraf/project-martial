#include <bits/stdc++.h>

using namespace std;

int main() {
  int n;
  cin >> n;
  int a[n], b[n];
  int max = 0;
  int count = 0;
  for (int i = 0; i < n; i++) {
    cin >> a[i];
    count -= a[i];
    // cout << count << endl;
    if (max < count) {
      // cout << max << endl;
      max = count;
    }
    cin >> b[i];
    count += b[i];
    // cout << count << endl;
    if (max < count) {
      // cout << max << endl;
      max = count;
    }
  }

  cout << max << endl;
  return 0;
}