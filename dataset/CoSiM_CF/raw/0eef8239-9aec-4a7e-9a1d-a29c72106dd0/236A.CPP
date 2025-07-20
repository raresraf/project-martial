// https://codeforces.com/contest/236/problem/A

/* https://codeforces.com/problemset/problem/236/A */

#include <iostream>

#include <string>

#include <bits/stdc++.h>

using namespace std;

int main()

{

  string s;

  cin >> s;

  int n[26];

  for (int i = 0; i < 26; i++)

  {

    n[i] = 0;
  }

  for (int i = 0; i < s.length(); i++)

  {

    int temp = s[i] - 'a';

    n[temp] = 1;
  }

  int count = 0;

  for (int i = 0; i < 26; i++)

  {

    if (n[i] > 0)

      count++;
  }

  // cout << count << endl;

  if (count % 2 == 1)

    cout << "IGNORE HIM!";

  else

    cout << "CHAT WITH HER!";
}