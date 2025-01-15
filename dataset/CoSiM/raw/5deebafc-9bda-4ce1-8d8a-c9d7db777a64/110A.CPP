// https://codeforces.com/contest/110/problem/A

/* https://codeforces.com/problemset/problem/110/A */

#include <iostream>

#include <string>

#include <bits/stdc++.h>

using namespace std;

int main()

{

  long long int n;

  cin >> n;

  stringstream ss;

  ss << n;

  string s = ss.str();

  // cout << s << endl;

  // string s = to_string(n);

  int lucky = 0;

  for (int i = 0; i < s.length(); i++)

  {

    if (s[i] == '4' || s[i] == '7')

      lucky++;
  }

  if (lucky == 7 or lucky == 4)

  {

    cout << "YES";

  }

  else

  {

    cout << "NO";
  }
}