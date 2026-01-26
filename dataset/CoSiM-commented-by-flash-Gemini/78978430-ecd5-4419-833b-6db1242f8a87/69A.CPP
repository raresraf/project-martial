#include <bits/stdc++.h>
using namespace std;
int main() {
  int n;
  cin >> n;
  int t1, t2, t3, f1, f2, f3;
  f1 = f2 = f3 = 0;
  for (int i = 0; i < n; i++) {
    cin >> t1 >> t2 >> t3;
    f1 += t1;
    f2 += t2;
    f3 += t3;
  }
  if (f1 || f2 || f3) {
    cout << "NO" << endl;
  } else {
    cout << "YES" << endl;
  }
  return 0;
}
