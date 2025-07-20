// https://codeforces.com/contest/281/problem/A

/* https://codeforces.com/problemset/problem/281/A */

#include <iostream>

#include <string>

#include <bits/stdc++.h>

using namespace std;

int main()

{

  string s;

  cin >> s;

  if (int(s[0]) > 96)

    s[0] = s[0] - 'a' + 'A';

  cout << s;
}