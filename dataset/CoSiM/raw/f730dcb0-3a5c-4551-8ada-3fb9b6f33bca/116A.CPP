// https://codeforces.com/contest/116/problem/A

/* https://codeforces.com/problemset/problem/116/A */

#include <iostream>

#include <string>

#include <bits/stdc++.h>

using namespace std;

int main()

{

  int n;

  cin >> n;

  int a[n], b[n];

  for (int i = 0; i < n; i++)

  {

    cin >> a[i] >> b[i];
  }

  int max = 0, cap = 0;

  for (int i = 0; i < n; i++)

  {

    cap += b[i] - a[i];

    // cout << "For iteration " << i << " cap is " << cap << endl;

    if (max < cap)

      max = cap;
  }

  cout << max;
}