#include <bits/stdc++.h>

using namespace std;
bool td = false;
int main() {
  int n;
  cin >> n;

  for (int i(2); i < n; i++) {
    if (i % 2 == 0) {
      if ((n - i) % 2 == 0) {
        td = true;
      }
    } else {
      continue;
    }
  }

  if (td) {
    cout << "YES" << '\n';
  } else {
    cout << "NO" << '\n';
  }

  return 0;
}
