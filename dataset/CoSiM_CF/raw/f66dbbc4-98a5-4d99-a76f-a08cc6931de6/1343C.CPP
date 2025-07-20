// https://codeforces.com/contest/1343/problem/C

#include <iostream>

#include <bits/stdc++.h>

using namespace std;

int sign(int n)

{

  if (n == 0)
    return 0;

  if (n > 0)
    return 1;

  return -1;
}

int main()

{

  int t;

  cin >> t;

  while (t--)

  {

    int n;

    cin >> n;

    int a[n];

    int sign_prev = 0;

    int prev_val = 0;

    int temp;

    long long int ans = 0;

    for (int i = 0; i < n; i++)

    {

      cin >> a[i];

      if (i == 0)

      {

        ans += a[i];

        sign_prev = sign(a[i]);

        prev_val = a[i];

      }

      else

      {

        temp = sign(a[i]);

        if (temp == sign_prev)

        {

          ans = ans - prev_val + max(prev_val, a[i]);

          prev_val = max(prev_val, a[i]);

        }

        else

        {

          ans += a[i];

          prev_val = a[i];

          sign_prev = temp;
        }
      }
    }

    cout << ans << endl;
  }
}