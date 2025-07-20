#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <numeric> // accumulate(v.begin(), v.end(), 0)
#include <vector>

using namespace std;
#define ll long long

int main() {
  int n;
  cin >> n;
  vector<int> A(n);
  bool isInfinite = false;
  int prev = 1;
  for (int i = 0; i < n; i++) {
    cin >> A[i];
    if ((A[i] == 2 && prev == 3) || (A[i] == 3 && prev == 2))
      isInfinite = true;
    prev = A[i];
  }
  if (isInfinite)
    cout << "Infinite" << endl;
  else {
    prev = A[0];
    ll ans = 0;
    for (int i = 1; i < n; i++) {
      if (prev == 1) {
        if (A[i] == 2) {
          if (i > 1 && A[i - 2] == 3)
            ans += 2;
          else
            ans += 3;
        } else {
          ans += 4;
        }
      } else if (prev == 2) {
        ans += 3;
      } else {
        ans += 4;
      }
      prev = A[i];
    }
    cout << "Finite" << endl;
    cout << ans << endl;
  }
  return 0;
}
