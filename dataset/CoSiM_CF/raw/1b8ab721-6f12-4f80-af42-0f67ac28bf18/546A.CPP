#include <bits/stdc++.h>
using namespace std;
int main(void) {
  // your code goes here
  long long int n, k, w;
  cin >> k >> n >> w;
  long long int ans = k * ((w * (w + 1)) / 2) - n;
  if (ans <= 0)
    cout << 0 << endl;
  else
    cout << ans << endl;
  return 0;
}