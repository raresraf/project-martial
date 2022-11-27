// https://codeforces.com/contest/4/problem/A

#include <iostream>

#include <bits/stdc++.h>

using namespace std;

int main()

{

  int w;

  cin >> w;

  if ((w % 2 == 0) && (w >= 4))
    cout << "YES" << endl;

  else
    cout << "NO" << endl;
}