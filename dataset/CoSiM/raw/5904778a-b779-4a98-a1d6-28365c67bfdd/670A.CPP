#include <bits/stdc++.h>
using namespace std;

long int n;

int main() {
  int min, max;
  cin >> n;
  if (n <= 7) {
    if (n == 1)
      cout << "0"
           << " "
           << "1";

    else
      cout << "0"
           << " "
           << "2";

  } else {
    int rem = n % 7;
    int res = n / 7;
    if (rem < 5) {
      min = res * 2;
      if (rem == 0)
        max = res * 2;
      else
        max = res * 2 + rem;
      cout << min << " " << max;
    } else {
      min = res * 2 + rem;
      max = res * 2 + 2;
      cout << min << " " << max;
    }
  }

  return 0;
}
