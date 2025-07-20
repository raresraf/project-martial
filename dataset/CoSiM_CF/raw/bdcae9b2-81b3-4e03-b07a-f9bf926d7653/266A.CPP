// https://codeforces.com/contest/266/problem/A

/* https://codeforces.com/problemset/problem/266/A */

#include <iostream>

#include <string>

#include <bits/stdc++.h>

using namespace std;

int main()

{

  int n;

  cin >> n;

  string s;

  cin >> s;

  int stone = 0;

  for (int i = 0; i < n; i++)

  {

    if (s[i] == s[i + 1])

      stone++;
  }

  cout << stone;
}
