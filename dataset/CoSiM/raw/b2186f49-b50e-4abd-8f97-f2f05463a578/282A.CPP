// https://codeforces.com/contest/282/problem/A

/* https://codeforces.com/problemset/problem/282/A */

#include <iostream>

#include <string>

#include <bits/stdc++.h>

using namespace std;

int main()

{

  int n;

  int count = 0;

  cin >> n;

  string str, str1, str2, str3, str4;

  str1 = "X++";

  str2 = "++X";

  str3 = "X--";

  str4 = "--X";

  for (int i = 0; i < n; i++)

  {

    cin >> str;

    if ((str.compare(str1) == 0) || (str.compare(str2) == 0))

      count++;

    else

      count--;
  }

  cout << count;
}