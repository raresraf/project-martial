// https://codeforces.com/contest/71/problem/A

/* https://codeforces.com/problemset/problem/71/A */

#include <iostream>

#include <string>

#include <bits/stdc++.h>

using namespace std;

int main()

{

  int n;

  cin >> n;

  string s[n];

  for (int i = 0; i < n; i++)

  {

    cin >> s[i];
  }

  string temp1, temp2 = "";

  for (int i = 0; i < n; i++)

  {

    temp1 = s[i];

    if (temp1.length() > 10)

    {

      temp2 = "";

      string temp3(1, temp1[0]);

      string temp4(1, temp1[temp1.length() - 1]);

      stringstream ss;

      ss << (temp1.length() - 2);

      string temp5 = ss.str();

      temp2 = temp2 + temp3 + temp5 + temp4;

      // cout << temp2;

      s[i] = temp2;
    }

    cout << s[i] << endl;
  }
}