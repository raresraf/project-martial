#include <bits/stdc++.h>
using namespace std;
void solve() {
  int k, w;    // k la gia cua qua chuoi dau tien, w la so qua chuoi
  long long n; // n la so tien a ta co
  cin >> k >> n >> w;
  int s = 0;
  for (int i = 1; i <= w; i++) {
    s += i;
  }
  long long kq = k * s - n;
  if (kq > 0)
    cout << kq;
  else
    cout << 0;
}
int main() {

  solve();
  return 0;
}
