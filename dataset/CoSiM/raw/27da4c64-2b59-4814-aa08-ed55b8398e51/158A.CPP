// https://codeforces.com/contest/158/problem/A

#include <iostream>

#include <bits/stdc++.h>

using namespace std;

int main()

{

  int n, k;

  cin >> n >> k;

  int temp;

  int a[n];

  int ct = 0;

  for (int i = 0; i < n; i++)

  {

    cin >> a[i];

    if ((i < k) && (a[i] > 0))
      ct++;
  }

  int val = a[k - 1];

  for (int i = k; i < n; i++)
    if ((a[i] == val) && (a[i] > 0))
      ct++;

  cout << ct << endl;
}