#include <bits/stdc++.h>
using namespace std;

int main() {
  int n, k, a, current;
  int result = 0;
  cin >> n >> k;
  for (int i = 0; i < n; i++) {
    cin >> a;
    if (a == 0) {
      break;
    }
    if (result < k) {
      result++;
      current = a;
    } else {
      if (current == a) {
        result++;
      } else {
        break;
      }
    }
  }
  cout << result;
  return 0;
}
