#include <bits/stdc++.h>
using namespace std;

int main(int argc, char *argv[]) {
  int n;
  cin >> n;

  int ans = 0;
  for (int j = 0; j < n; ++j) {
    int y = 0;

    int x;
    for (int i = 0; i < 3; ++i) {
      cin >> x;
      y += x;
    }
    if (y >= 2)
      ++ans;
  }

  cout << ans << endl;
}
