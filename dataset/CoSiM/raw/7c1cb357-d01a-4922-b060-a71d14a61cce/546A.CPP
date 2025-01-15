#include <bits/stdc++.h>

using namespace std;

int main() {
  int k, n, w;
  cin >> k >> n >> w;
  int count = 0;
  for (int i = 0; i < w; ++i) {
    count += (i + 1) * k;
  }
  if (n >= count) {
    cout << 0 << endl;
  } else {
    cout << count - n << endl;
  }
}
