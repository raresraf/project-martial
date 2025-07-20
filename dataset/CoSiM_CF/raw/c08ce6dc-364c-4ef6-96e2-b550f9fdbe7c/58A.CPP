#include <bits/stdc++.h>

using namespace std;

int main()

{

  long long int n, l, r, d;

  cin >> n;

  for (long long int i = 0; i < n; i++)

  {

    cin >> l >> r >> d;

    if (d < l)
      cout << d << endl;

    else if (d >= l && d <= r) {

      long long int a = r / d;

      cout << (a + 1) * d << endl;

    }

    else
      cout << d << endl;
  }
}