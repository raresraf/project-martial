#include <bits/stdc++.h>
using namespace std;

int main() {
  int n;
  cin >> n;
  int a[n], b[n], left;
  vector<int> total;
  for (int i = 0; i < n; i++) {
    cin >> a[i] >> b[i];
  }
  if (a[0] == 0 && b[n - 1] == 0) {
    total.push_back(b[0]);
    for (int i = 1; i < (n - 1); i++) {
      left = b[i - 1] - a[i];
      int x = left + b[i];
      b[i] = x;
      total.push_back(x);
    }

    sort(total.begin(), total.end());
    // int m= total.end()-1;
    vector<int>::iterator itr = total.end();
    cout << *(itr - 1) << endl;
  }
  return 0;
}
