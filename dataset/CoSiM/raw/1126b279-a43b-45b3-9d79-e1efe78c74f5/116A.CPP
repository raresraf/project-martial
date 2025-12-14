#include <bits/stdc++.h>
using namespace std;
int main() {
  int t;
  cin >> t;
  int maxCount = 0, count = 0;
  while (t--) {
    int a, b;
    cin >> a >> b;
    count -= a;
    count += b;
    if (count > maxCount) {
      maxCount = count;
    }
  }
  cout << maxCount << endl;
}
// iu
