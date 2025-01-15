#include <bits/stdc++.h>
#include <string.h>
using namespace std;
int main() {
  ios::sync_with_stdio(0);
  cin.tie(0);
  cout.tie(0);
  int n, counts = 0;
  int petya, vasaya, tonya;
  cin >> n;
  for (int i = 0; i < n; i++) {
    cin >> petya;
    cin >> vasaya;
    cin >> tonya;
    if ((petya == 1 && vasaya == 1 && tonya == 1) ||
        (petya == 1 && vasaya == 1 && tonya == 0) ||
        (petya == 1 && vasaya == 0 && tonya == 1) ||
        (petya == 0 && vasaya == 1 && tonya == 1)) {
      counts = counts + 1;
    } else {
      counts = counts + 0;
    }
  }
  cout << counts;

  return 0;
}
