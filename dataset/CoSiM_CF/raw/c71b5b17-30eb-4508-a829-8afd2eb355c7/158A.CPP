#include <bits/stdc++.h>

using namespace std;

int main() {
  int n, k, b, ct = 0;
  vector<int> a;
  cin >> n >> k;
  for (int i = 0; i < n; i++) {
    cin >> b;
    a.push_back(b);
  }
  for (int i = 0; i < n; i++) {
    if (a[k - 1] <= a[i] && a[i] > 0)
      ct++;
    if (a[i] < a[k - 1])
      break;
  }
  cout << ct << endl;
  return 0;
}
