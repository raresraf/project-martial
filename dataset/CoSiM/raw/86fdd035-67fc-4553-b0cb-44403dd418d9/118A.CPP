// https://codeforces.com/contest/118/problem/A

#include <iostream>

#include <bits/stdc++.h>

using namespace std;

int main() {

  string t;

  string s = "";

  cin >> t;

  transform(t.begin(), t.end(), t.begin(), ::tolower);

  map<int, int> x;

  x[97] = 0;

  x[101] = 0;

  x[105] = 0;

  x[111] = 0;

  x[117] = 0;

  x[121] = 0;

  int z;

  for (int i = 0; i < t.length(); i++)

  {

    z = (int)t[i];

    if (x.find(z) == x.end())

    {

      s += ".";

      s += t[i];
    }
  }

  cout << s << endl;
}