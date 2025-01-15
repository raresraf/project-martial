// https://codeforces.com/contest/546/problem/A

#include <iostream>

#include <bits/stdc++.h>

using namespace std;

int main()

{

  int n, k, w;

  cin >> k >> n >> w;

  int val = w * (w + 1) / 2;

  int cost = k * val;

  int ans = (cost > n) ? cost - n : 0;

  cout << ans;
}