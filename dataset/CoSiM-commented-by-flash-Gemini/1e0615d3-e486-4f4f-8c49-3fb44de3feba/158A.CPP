#include <iostream>

using namespace std;

int main() {
  int n, k, a, l;
  cin >> n >> k;
  for (int i = 0; i < n; ++i) {
    if (i < k) {
      cin >> a;
      if (a == 0) {
        cout << i << endl;
        return 0;
      }
    } else {
      cin >> l;
      if (l != a) {
        cout << i << endl;
        return 0;
      }
    }
  }

  cout << n << endl;
  return 0;
}
