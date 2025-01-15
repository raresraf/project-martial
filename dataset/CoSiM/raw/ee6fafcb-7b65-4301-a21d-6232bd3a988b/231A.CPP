// https://codeforces.com/contest/231/problem/A

/* https://codeforces.com/problemset/problem/231/A */

#include <iostream>

#include <string>

#include <bits/stdc++.h>

using namespace std;

int main()

{

  int n;

  int x, y, z;

  cin >> n;

  int count = 0;

  for (int i = 0; i < n; i++)

  {

    cin >> x >> y >> z;

    if ((x + y + z) >= 2)

      count++;
  }

  cout << count;
}