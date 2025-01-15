#include <bits/stdc++.h>
using namespace std;

int main() {
  // freopen("input.txt", "r", stdin);

  int t = 0;
  cin >> t;
  for (int i = 0; i < t; ++i) {
    int n = 0;
    cin >> n;
    bool pos = false;
    int num = 0;
    int max = 0;
    int init = true;
    long long sum = 0;
    for (int j = 0; j < n; ++j) {
      cin >> num;
      if (init) {
        init = false;
        max = num;
        if (num > 0) {
          pos = true;
        }
      } else if ((num > 0 && !pos) || (num < 0 && pos)) {
        pos = !pos;
        sum += (long long)max;
        max = num;
      } else {
        if (num > max) {
          max = num;
        }
      }
    }
    sum += (long long)max;
    cout << sum << "\n";
  }
  return 0;
}
