#include <bits/stdc++.h>
using namespace std;
int main() {
  long long int n;
  int t;
  int cnt = 0;
  cin >> n;
  while (n) {
    t = n % 10;
    if (t == 4 || t == 7) {
      cnt++;
    }
    n /= 10;
  }
  if (cnt == 4 || cnt == 7) {
    cout << "YES" << endl;
  } else {
    cout << "NO" << endl;
  }
  return 0;
}
