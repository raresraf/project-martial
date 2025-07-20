#include <bits/stdc++.h>

using namespace std;

int main() {

  int x;

  cin >> x;

  for (int i = 0; i < x; i++) {

    long long int a, b, c;

    long long int k = 1000000000000000001;

    long long int s = 0;

    cin >> a >> b >> c;

    s = s + (a + b + c);

    k = min(a, k);

    k = min(b, k);

    if (k > c) {

      if (s / 3 > k) {

        cout << k << endl;

      }

      else {

        cout << s / 3 << endl;
      }

    }

    else if (s >= 3 * k) {

      cout << k << endl;
    }
  }
}