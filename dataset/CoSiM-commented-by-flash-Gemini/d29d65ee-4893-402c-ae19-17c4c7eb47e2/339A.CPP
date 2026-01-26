// https://codeforces.com/contest/339/problem/A

/* https://codeforces.com/problemset/problem/339/A */

#include <iostream>

#include <string>

#include <bits/stdc++.h>

using namespace std;

int main()

{

  string s1;

  cin >> s1;

  int one = 0, two = 0, three = 0;

  for (int i = 0; i < s1.length(); i++)

  {

    if (s1[i] == '1')

      one++;

    if (s1[i] == '2')

      two++;

    if (s1[i] == '3')

      three++;
  }

  string s2 = "";

  if (one > 0)

  {

    for (int i = 0; i < one; i++)

    {

      s2 = s2 + "1+";
    }
  }

  if (two > 0)

  {

    for (int i = 0; i < two; i++)

    {

      s2 = s2 + "2+";
    }
  }

  if (three > 0)

  {

    for (int i = 0; i < three; i++)

    {

      s2 = s2 + "3+";
    }
  }

  string s3 = s2.substr(0, s2.size() - 1);

  cout << s3;
}