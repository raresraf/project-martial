// https://codeforces.com/contest/266/problem/B

/* https://codeforces.com/problemset/problem/266/B */

#include <iostream>

#include <string>

#include <cstdlib>

using namespace std;

int main()

{

  int n, t;

  string s;

  char temp;

  cin >> n >> t;

  cin >> s;

  for (int i = 0; i < t; i++)

  {

    for (int j = 0; j < n; j++)

    {

      if (s[j] == 'B' && s[j + 1] == 'G')

      {

        s[j] = 'G';

        s[j + 1] = 'B';

        j++;
      }
    }
  }

  cout << s;
}