#include <bits/stdc++.h>
using namespace std;

#define DBG(x) cout << #x << " = " << (x) << "\n"
#define END(e)                                                                 \
  {                                                                            \
    cout << (e) << "\n";                                                       \
    return EXIT_SUCCESS;                                                       \
  }

int main() {
  ios::sync_with_stdio(false);
  cin.tie(nullptr);
  int v;
  for (int r = 1; r < 6; ++r) {
    for (int c = 1; c < 6; ++c) {
      cin >> v;
      if (v == 1)
        END(abs(r - 3) + abs(c - 3))
    }
  }
}
