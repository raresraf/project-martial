// https://codeforces.com/contest/160/problem/A

#include <iostream>

#include <bits/stdc++.h>

using namespace std;

int main()

{

  int n;

  cin >> n;

  int a[n];

  for (int i = 0; i < n; i++)
    cin >> a[i];

  sort(a, a + n);

  int count = 0, sum = 0, remsum = 0;

  for (int i = 0; i < n; i++)
    remsum += a[i];

  int i = n - 1;

  while ((i >= 0) && (sum <= remsum))

  {

    sum += a[i];

    remsum -= a[i];

    count++;
    i--;
  }

  cout << count;
}