// problem 0
#include <iostream>
#define ll long long int
using namespace std;
int main() {
  ll n;
  ll min = 0;
  ll max = 0;
  cin >> n;
  if (n == 0) {
    min = 0;
    max = 0;
  } else if (n == 1) {
    min = 0;
    max = 1;
  } else if (n < 6 && n != 1 && n != 0) {
    min = 0;
    max = 2;
  } else if (n == 6) {
    min = 1;
    max = 2;
  } else if (n == 7) {
    min = 2;
    max = 2;
  } else {
    ll ans = n / 7;
    ll rem = n % 7;
    if (rem == 1) {
      min = 2 * ans;
      max = 2 * ans + 1;
    } else if (rem == 6) {
      min = 2 * ans + 1;
      max = 2 * ans + 2;
    } else if (rem != 0 && rem != 1) {
      min = 2 * ans;
      max = 2 * ans + 2;
    } else if (rem == 0) {
      min = 2 * ans;
      max = 2 * ans;
    }
  }
  cout << min << " " << max;
  return 0;
}
