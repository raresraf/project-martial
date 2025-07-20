#include <bits/stdc++.h>
using namespace std;

int main() {
  int n, k, i, c = 0, num;
  vector<int> a;
  cin >> n >> k;
  for (i = 0; i < n; i++) {
    cin >> num;
    a.push_back(num);
  }
  for (i = 0; i < n; i++) {
    if ((a[i] >= a[k - 1]) && (a[i] > 0)) {
      c++;
    }
  }
  cout << c << endl;
  return 0;
}
