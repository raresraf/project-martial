#include <bits/stdc++.h>
using namespace std;
int main() {
  long long int n, k, w, cnt = 0;
  cin >> k >> n >> w;
  for (int i = 1; i <= w; i++) {
    cnt += (i * k);
  }
  cnt -= n;
  if (cnt <= 0) {
    cout << 0;
  } else {
    cout << cnt;
  }
  return 0;
}
