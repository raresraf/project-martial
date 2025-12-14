// https://codeforces.com/contest/112/problem/A

/* https://codeforces.com/problemset/problem/112/A */

#include <iostream>

#include <string>

#include <bits/stdc++.h>

using namespace std;

int main()

{

  string str1, str2;

  cin >> str1;

  // cout << str1;

  cin >> str2;

  // cout << str2;

  int comp;

  transform(str1.begin(), str1.end(), str1.begin(), ::tolower);

  transform(str2.begin(), str2.end(), str2.begin(), ::tolower);

  comp = str1.compare(str2);

  if (comp != 0)

    comp = comp / (abs(comp));

  cout << comp;
}