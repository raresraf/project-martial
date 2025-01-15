#include <bits/stdc++.h>

#define ll long long int

#define mod 1000000007

#define pb push_back

#define mp make_pair

#define ff first

#define ss second

#define all(v) v.begin(), v.end()

using namespace std;

int main()

{

  ios::sync_with_stdio(false);

  cin.tie(0);

  int st = 0;

  string s;

  cin >> s;

  if (s[0] - '0')

    st = 1;

  int cnt = 1, fin = 0;

  for (int i = 1; i < s.length(); i++)

  {

    if ((s[i] - '0') == st)

      cnt++;

    else

    {

      if (st)

        st = 0;

      else

        st = 1;

      fin = max(fin, cnt);

      cnt = 1;
    }
  }

  fin = max(fin, cnt);

  if (fin >= 7)

    cout << "YES";

  else

    cout << "NO";

  return 0;
}