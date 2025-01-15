#include <iostream>
using namespace std;

int main() {
  int n, k, a, t, count;
  cin >> n >> k;
  k--;
  count = 0;
  for (int i = 0; i < n; i++) {
    if (i < k) {
      cin >> a;
      if (a > 0)
        count++;
    } else if (i == k) {
      cin >> t;
      if (t > 0)
        count++;
    } else {
      cin >> a;
      if (a == t && t > 0)
        count++;
    }
  }
  cout << count;

  return 0;
}