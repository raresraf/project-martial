#include <bits/stdc++.h>
using namespace std;
typedef long long int ll;
int main() {
  ll a;
  cin >> a;
  ll min = 0, max = 0;
  ll k = a % 7;
  if (a == 1) {
    max = 1;
    min = 0;
  } else if (a >= 2 && a <= 5) {
    max = 2;
    min = 0;
  } else if (a == 6) {
    max = 2;
    min = 1;
  } else if (a == 7) {
    max = 2;
    min = 2;
  } else {
    min = (a / 7) * 2;
    max = min;
    if (a % 7 == 1)
      max++;
    else if (a % 7 >= 2 && a % 7 <= 5)
      max += 2;
    else if (a % 7 == 6) {
      min++;
      max += 2;
    }
  }
  cout << min << " " << max << endl;
  return 0;
}