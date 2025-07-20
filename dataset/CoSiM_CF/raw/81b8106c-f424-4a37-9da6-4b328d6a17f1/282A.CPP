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
  int n, x = 0;
  char aux, ch;
  cin >> n;
  for (int i = 0; i < n; ++i) {
    cin >> aux >> ch >> aux;
    if (ch == '-')
      --x;
    else
      ++x;
  }
  END(x)
}
